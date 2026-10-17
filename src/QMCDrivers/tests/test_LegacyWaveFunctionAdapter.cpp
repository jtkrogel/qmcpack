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
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "Particle/ParticleSet.h"
#include "Particle/ParticleSetPool.h"
#include "Particle/VirtualParticleSet.h"
#include "OhmmsData/Libxml2Doc.h"
#include "QMCDrivers/WFTrain/EnergyGradientAccumulator.h"
#include "QMCDrivers/WFTrain/FirstOrderOptimizer.h"
#include "QMCDrivers/WFTrain/HighParameterTraining.h"
#include "QMCDrivers/WFTrain/TrainingCheckpoint.h"
#include "QMCDrivers/WFTrain/TrainingNumerics.h"
#include "QMCDrivers/WFTrain/TrainingWaveFunctionBinding.h"
#include "QMCDrivers/WFTrain/SampledLinearMethod.h"
#include "QMCDrivers/WFTrain/AcceleratedLinearMethod.h"
#include "QMCHamiltonians/BareKineticEnergy.h"
#include "QMCHamiltonians/QMCHamiltonian.h"
#include "QMCHamiltonians/ECPComponentBuilder.h"
#include "QMCHamiltonians/ECPotentialBuilder.h"
#include "QMCHamiltonians/NonLocalECPComponent.h"
#include "QMCHamiltonians/LocalECPotential.h"
#include "QMCHamiltonians/CoulombPBCAA.h"
#include "QMCHamiltonians/CoulombPotential.h"
#include "QMCWaveFunctions/ElectronGas/FreeOrbital.h"
#include "ResourceCollection.h"
#include "Utilities/StdRandom.h"
#include "Utilities/FakeRandom.h"
#include "LegacyP1MatrixTest.h"
#include "QMCWaveFunctions/Optimization/LegacyWaveFunctionAdapter.h"
#include "QMCWaveFunctions/Jastrow/BsplineFunctor.h"
#include "QMCWaveFunctions/Jastrow/TwoBodyJastrow.h"
#include "QMCWaveFunctions/Jastrow/J1OrbitalSoA.h"
#include "QMCWaveFunctions/Jastrow/eeI_JastrowBuilder.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCWaveFunctions/WaveFunctionFactory.h"
#include "QMCWaveFunctions/Fermion/MultiSlaterDetTableMethod.h"
#include "QMCWaveFunctions/Fermion/SlaterDet.h"
#include "QMCWaveFunctions/Fermion/DiracDeterminant.h"
#include "QMCWaveFunctions/RotatedSPOs.h"
#include "QMCWaveFunctions/LCAO/LCAOrbitalSet.h"
#include "QMCWaveFunctions/LCAO/LCAOrbitalBuilder.h"
#include "QMCWaveFunctions/WaveFunctionComponent.h"
#include "SimulationCell.h"
#include "io/hdf/hdf_archive.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <limits>
#include <iomanip>
#include <sstream>
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

/// Add the qualified eight-coordinate polynomial functor; default spin slots alias.
void addPolynomialJ3(TrialWaveFunction& wavefunction, ParticleSet& particles, ParticleSet& ions)
{
  Libxml2Document input;
  REQUIRE(input.parseFromString(R"(<jastrow name="J3" type="eeI" function="polynomial" source="ion0">
    <correlation ispecies=")" + ions.getSpeciesSet().getSpeciesName(0) + R"(" especies1="u" especies2="u"
                 isize="2" esize="2" rcut="4">
      <coefficients id="j3" type="Array" optimize="yes">
        0.000001 -0.000002 0.000003 -0.000004 0.000002 -0.000001 0.000003 -0.000002
      </coefficients>
    </correlation></jastrow>)"));
  eeI_JastrowBuilder builder(OHMMS::Controller, particles, ions);
  wavefunction.addComponent(builder.buildComponent(input.getRoot()));
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
  std::size_t prepareCalls() const noexcept { return prepare_calls_; }
  LegacyWaveFunctionReplaySample prepareSample(std::size_t index) override
  {
    ++prepare_calls_;
    particles_.R[0] = {0.2 + shifts_.at(index), -0.3, 0.4};
    particles_.R[1] = {-0.5, 0.1 + shifts_.at(index), 0.7};
    particles_.update();
    return {wavefunction_, particles_};
  }
private:
  TrialWaveFunction& wavefunction_;
  ParticleSet& particles_;
  std::vector<double> shifts_;
  std::size_t prepare_calls_ = 0;
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

TEST_CASE("Legacy fixed-Slater ECP products reuse the accepted physical grid",
          "[drivers][training][legacy_adapter][ecp]")
{
#ifdef QMC_COMPLEX
  SKIP("This gate qualifies real CPU fixed-Slater/Jastrow ECP products");
#endif
  const bool polynomial_j3 = GENERATE(false, true);
  CAPTURE(polynomial_j3);
  ParticleSetPool pool(OHMMS::Controller);
  auto ion_owner = std::make_unique<ParticleSet>(pool.getSimulationCell());
  auto electron_owner = std::make_unique<ParticleSet>(pool.getSimulationCell());
  auto& ions = *ion_owner;
  auto& particles = *electron_owner;
  ions.setName("ion0");
  ions.create({1});
  ions.R[0] = {0, 0, 0};
  auto& ion_species = ions.getSpeciesSet();
  const int be = ion_species.addSpecies("Be");
  const int ion_charge = ion_species.addAttribute("charge");
  const int atomic_number = ion_species.addAttribute("atomic_number");
  ion_species(ion_charge, be) = 2;
  ion_species(atomic_number, be) = 4;
  ions.resetGroups();
  particles.setName("e");
  particles.create({1, 1});
  particles.R[0] = {0.2, -0.3, 0.4};
  particles.R[1] = {-0.5, 0.1, 0.7};
  auto& species = particles.getSpeciesSet();
  const int up = species.addSpecies("u"), down = species.addSpecies("d");
  const int mass = species.addAttribute("mass"), charge = species.addAttribute("charge");
  species(mass, up) = species(mass, down) = 1;
  species(charge, up) = species(charge, down) = -1;
  particles.resetGroups();
  pool.addParticleSet(std::move(ion_owner));
  pool.addParticleSet(std::move(electron_owner));
  Libxml2Document definition;
  REQUIRE(definition.parseFromString(R"(
    <wavefunction name="psi0" target="e">
      <sposet_collection type="MolecularOrbital" source="ion0">
        <basisset keyword="STO" transform="no">
          <atomicBasisSet type="STO" elementType="Be" normalized="no">
            <basisGroup rid="R0" l="0" m="0" type="Slater"><radfunc n="1" exponent="1.2"/></basisGroup>
          </atomicBasisSet>
        </basisset>
        <sposet name="fixed" size="1"><coefficient size="1" id="identity">1</coefficient></sposet>
      </sposet_collection>
      <determinantset><slaterdeterminant><determinant sposet="fixed"/><determinant sposet="fixed"/></slaterdeterminant></determinantset>
      <jastrow name="J2" type="Two-Body" function="Bspline">
        <correlation rcut="4" size="4" speciesA="u" speciesB="d">
          <coefficients id="ud" type="Array" optimize="yes">0.12 -0.08 0.04 -0.01</coefficients>
        </correlation>
      </jastrow>
    </wavefunction>)"));
  WaveFunctionFactory factory(particles, pool.getPool(), OHMMS::Controller);
  auto source = factory.buildTWF(definition.getRoot(), RuntimeOptions{});
  REQUIRE(source);
  if (polynomial_j3) addPolynomialJ3(*source, particles, ions);
  ions.update();
  particles.update();
  source->evaluateLog(particles);
  const auto initial_positions = particles.R;
  const auto initial_ions = ions.R;
  const auto initial_log = source->getLogPsi();
  ParticleSet sibling_particles(particles);
  auto sibling = source->makeClone(sibling_particles);
  sibling_particles.update();
  sibling->evaluateLog(sibling_particles);
  const auto sibling_log = sibling->getLogPsi();
  ParticleSet private_particles(particles);
  auto private_wf = source->makeClone(private_particles);
  private_particles.update();
  auto replay = std::make_shared<ShiftedLegacyReplay>(*private_wf, private_particles, std::vector<double>{0});
  LegacyWaveFunctionAdapter adapter("conventional", *source, adapterOptions());
  const auto snapshot = std::make_shared<const StructuredParameterSnapshot>(adapter.snapshotParameters());
  REQUIRE(snapshot->values.size() == (polynomial_j3 ? 12 : 4));
  auto op = adapter.makeDerivativeOperator(replay, snapshot, 3, 7);
  auto consumer = adapter.makeNonLocalECPDerivativeConsumer(replay, snapshot, 3, 7);
  const std::vector<DerivativeValue> unit{1};
  const auto ecp_mask = localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);
  const auto total_mask = ecp_mask | localEnergyTermBit(LocalEnergyTerm::KINETIC);
  const VJPCoefficientChannel channel{"ecp", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(*op, unit), ecp_mask};
  RecordingParameterSink idle_sink;
  ECPComponentBuilder builder("Be", OHMMS::Controller, 2, 1);
  const auto asset = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() /
      "tests/pseudopotentials_for_tests/Be.BFD.xml";
  REQUIRE(builder.read_pp_file(asset.string()));
  REQUIRE(builder.pp_nonloc);
  auto nonlocal = std::make_unique<NonLocalECPotential>(ions, particles, false, true);
  auto* nonlocal_ptr = nonlocal.get();
  nonlocal->addComponent(0, std::move(builder.pp_nonloc));
  auto local = std::make_unique<LocalECPotential>(ions, particles);
  auto* local_ptr = local.get();
  local->add(0, std::move(builder.pp_loc), builder.Zeff);
  QMCHamiltonian ham;
  ham.addOperator(std::make_unique<BareKineticEnergy>(particles), "Kinetic");
  ham.addOperator(std::move(local), "LocalECP");
  ham.addOperator(std::move(nonlocal), "NonLocalECP");
  // Full Hamiltonian evaluation publishes each component through registered
  // observable/property slots, unlike direct isolated-operator evaluations.
  ham.addObservables(particles);
  StdRandom<QMCTraits::FullPrecRealType> rng(1931);
  nonlocal_ptr->setRandomGenerator(&rng);
  RefVectorWithLeader<ParticleSet> p_list(particles, {particles});
  RefVectorWithLeader<TrialWaveFunction> wf_list(*source, {*source});
  RefVectorWithLeader<QMCHamiltonian> h_list(ham, {ham});
  ResourceCollection p_resources("ecp_particles"), w_resources("ecp_wavefunctions"), h_resources("ecp_hamiltonians");
  particles.createResource(p_resources);
  source->createResource(w_resources);
  ham.createResource(h_resources);
  QMCHamiltonian::StreamingNonLocalECPResult accepted;
  {
    ResourceCollectionTeamLock<ParticleSet> p_lock(p_resources, p_list);
    ResourceCollectionTeamLock<TrialWaveFunction> w_lock(w_resources, wf_list);
    ResourceCollectionTeamLock<QMCHamiltonian> h_lock(h_resources, h_list);
    accepted = QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(
        h_list, wf_list, p_list, *consumer, {&channel, 1}, idle_sink);
  } // Release resources before the FD oracle creates independent private clones.
  const auto rng_before_actions = rng;
  CHECK(idle_sink.state() == DerivativeSinkState::IDLE);
  auto grid = consumer->releaseJVPReplay();
  REQUIRE(grid);
  CHECK_THROWS(consumer->releaseJVPReplay());
  CHECK(grid->gridFingerprint() == accepted.grid_fingerprint);
  REQUIRE(grid->points().size() > 0);
  CHECK(grid->evaluationStamps().empty());
  const std::vector<NonLocalECPReplayPoint> points(grid->points().begin(), grid->points().end());
  const auto grid_bytes = grid->storageDiagnostics().retained_numeric_bytes;
  op->bindNonLocalECPJVPReplay(std::move(grid));
  CHECK(op->capabilities().coversLocalEnergyTerms(total_mask));
  const auto storage = op->storageDiagnostics();
  // The adapter cap excludes component-owned J3 P*Ne caches and private clones.
  CHECK(storage.real_parameter_vectors == 2);
  CHECK(storage.retained_numeric_bytes >= storage.shared_workspace_bytes + grid_bytes);
  CHECK(storage.retained_numeric_bytes == storage.shared_workspace_bytes + storage.sample_position_bytes +
        storage.sample_auxiliary_bytes);
  CHECK(storage.real_parameter_vectors == 2);
  auto unbound = adapter.makeDerivativeOperator(replay, snapshot, 4, 8);
  CHECK(unbound->storageDiagnostics().shared_workspace_identity == storage.shared_workspace_identity);
  CHECK_FALSE(unbound->capabilities().coversLocalEnergyTerms(ecp_mask));
  const double local_value = local_ptr->evaluate(particles);
  // Independent finite-difference oracle recomputes complete ratios, but never
  // draws a grid or uses the captured accepted ratio at perturbed parameters.
  const auto fixed_grid_energy = [&](const StructuredParameterSnapshot& parameters) {
    ParticleSet test_particles(particles);
    auto test_wf = source->makeClone(test_particles);
    test_particles.update();
    adapter.installSnapshot(*test_wf, test_particles, parameters);
    const auto reference_log = test_wf->getLogPsi();
    DerivativeValue energy = kineticEnergy(test_particles) + local_value;
    for (const auto& point : points)
    {
      test_particles.R = initial_positions;
      for (int d = 0; d < 3; ++d) test_particles.R[point.electron][d] = point.replacement_position[d];
      test_particles.update();
      test_wf->evaluateLog(test_particles);
      energy += point.bare_weight * std::exp(test_wf->getLogPsi() - reference_log);
    }
    return energy;
  };
  const auto initial_energy = fixed_grid_energy(*snapshot);
  CHECK(initial_energy.real() == Catch::Approx(accepted.local_energies[0]).epsilon(2e-12).margin(2e-12));
  const VJPCoefficientChannel total{"total", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(*op, unit), total_mask};
  RecordingParameterSink transpose, hermitian;
  op->applyVJPs({&total, 1}, DerivativeAdjoint::TRANSPOSE, transpose);
  op->applyVJPs({&total, 1}, DerivativeAdjoint::HERMITIAN, hermitian);
  constexpr double step = 1e-5;
  std::vector<DerivativeValue> direction{0.5, -0.2, 0.3, -0.1};
  direction.resize(snapshot->values.size());
  for (std::size_t parameter = 4; parameter < direction.size(); ++parameter)
    direction[parameter] = 0.03 * (parameter - 3);
  DerivativeValue expected_direction{};
  double j3_response_norm = 0;
  for (std::size_t parameter = 0; parameter < snapshot->values.size(); ++parameter)
  {
    auto plus = *snapshot, minus = *snapshot;
    plus.values[parameter] += step;
    minus.values[parameter] -= step;
    const auto fd = (fixed_grid_energy(plus) - fixed_grid_energy(minus)) / (2 * step);
    CHECK(transpose.result()[0][parameter].real() == Catch::Approx(fd.real()).epsilon(3e-7).margin(3e-8));
    checkClose(hermitian.result()[0][parameter], std::conj(transpose.result()[0][parameter]));
    expected_direction += direction[parameter] * transpose.result()[0][parameter];
    if (parameter >= 4) j3_response_norm = std::hypot(j3_response_norm, transpose.result()[0][parameter].real());
  }
  if (polynomial_j3) REQUIRE(j3_response_norm > 1e-8);
  StructuredParameterVectorConstView vector(adapter.parameterSchema(), snapshot->version,
                                            {direction.data(), direction.size()});
  RecordingSampleSink first, second;
  op->applyLocalEnergyJVP(vector, total_mask, first);
  op->applyLocalEnergyJVP(vector, total_mask, second);
  checkClose(first.result()[0], expected_direction);
  CHECK(first.result() == second.result());
  auto expected_rng = rng_before_actions;
  CHECK(rng() == expected_rng());
  for (std::size_t electron = 0; electron < particles.R.size(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      CHECK(particles.R[electron][dimension] == initial_positions[electron][dimension]);
  for (int dimension = 0; dimension < 3; ++dimension)
    CHECK(ions.R[0][dimension] == initial_ions[0][dimension]);
  source->evaluateLog(particles);
  CHECK(source->getLogPsi() == initial_log);
  sibling->evaluateLog(sibling_particles);
  CHECK(sibling->getLogPsi() == sibling_log);
  OptVariables live_parameters;
  source->checkInVariables(live_parameters);
  live_parameters.resetIndex();
  REQUIRE(live_parameters.size_of_active() == snapshot->values.size());
  for (std::size_t i = 0; i < live_parameters.size(); ++i)
    if (live_parameters.where(i) >= 0)
      CHECK(live_parameters[i] == snapshot->values[live_parameters.where(i)]);
  CHECK(adapter.snapshotParameters().values == snapshot->values);

  const VJPCoefficientChannel score_only{"score", DerivativeProduct::SCORE_VJP, coefficientsFor(*op, unit), 0};
  const auto calls_before_score = replay->prepareCalls();
  RecordingParameterSink score_sink;
  op->applyVJPs({&score_only, 1}, DerivativeAdjoint::TRANSPOSE, score_sink);
  CHECK(replay->prepareCalls() == calls_before_score + 1); // No ECP replay in metric actions.
  RecordingSampleSink score_jvp;
  op->applyScoreJVP(vector, score_jvp);
  DerivativeValue score_expected{};
  for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
    score_expected += direction[parameter] * score_sink.result()[0][parameter];
  checkClose(score_jvp.result()[0], score_expected);

  // Exercise empty/ragged tiles, zero accepted ratios and stale binding without
  // a fresh energy traversal. Zero ratios have zero response ONLY because every
  // active parameter is a Jastrow and the fixed Slater node cannot move.
  const auto capture_tiles = [&](std::size_t version) {
    auto parameters = std::make_shared<StructuredParameterSnapshot>(*snapshot);
    parameters->version = version;
    auto captured = adapter.makeNonLocalECPDerivativeConsumer(replay, parameters, 4, 8);
    NonLocalECPDerivativeContext context{adapter.parameterSchema().providerId(), adapter.parameterSchema().fingerprint(),
        version, 4, 8, 1, points.size(), 456};
    const CoefficientView coefficients{adapter.parameterSchema().providerId(), adapter.parameterSchema().fingerprint(),
        version, 4, 8, {unit.data(), unit.size()}};
    const VJPCoefficientChannel coefficient{"ecp", DerivativeProduct::LOCAL_ENERGY_VJP, coefficients, ecp_mask};
    captured->begin(context, {&coefficient, 1}, idle_sink);
    std::size_t ordinal = 0;
    const std::vector<std::size_t> empty_offsets{0};
    const std::vector<VirtualParticleBatch::Segment> no_segments;
    const std::vector<ParticleSet::PosType> no_positions;
    VirtualParticleBatch empty(1, empty_offsets, no_segments, no_positions);
    captured->consume({empty, {}, {}, {}, ordinal++, 456});
    for (std::size_t i = 0; i < points.size(); ++i)
    {
      const auto& point = points[i];
      const std::vector<std::size_t> offsets{0, 1};
      const std::vector<VirtualParticleBatch::Segment> segments{{0, static_cast<int>(point.electron), true, 0}};
      std::vector<ParticleSet::PosType> positions(1);
      for (int d = 0; d < 3; ++d) positions[0][d] = point.replacement_position[d];
      VirtualParticleBatch tile(1, offsets, segments, positions);
      const QMCTraits::ValueType bare = point.bare_weight.real();
      const QMCTraits::ValueType ratio = i == 0 ? 0.0 : point.complete_ratio.real();
      captured->consume({tile, {&bare, 1}, {&ratio, 1}, {}, ordinal++, 456});
    }
    captured->end();
    return captured->releaseJVPReplay();
  };
  auto stale = capture_tiles(snapshot->version + 1);
  CHECK_THROWS(unbound->bindNonLocalECPJVPReplay(std::move(stale)));
  CHECK_FALSE(unbound->capabilities().coversLocalEnergyTerms(ecp_mask));
  unbound->bindNonLocalECPJVPReplay(capture_tiles(snapshot->version));
  CHECK(unbound->capabilities().coversLocalEnergyTerms(ecp_mask));
  const VJPCoefficientChannel zero_channel{"ecp", DerivativeProduct::LOCAL_ENERGY_VJP,
                                         coefficientsFor(*unbound, unit), ecp_mask};
  RecordingParameterSink zero_response;
  unbound->applyVJPs({&zero_channel, 1}, DerivativeAdjoint::TRANSPOSE, zero_response);
  // An independent ratio finite difference on precisely the retained nonzero knots.
  const auto nonzero_energy = [&](const StructuredParameterSnapshot& parameters) {
    ParticleSet test_particles(particles);
    auto test_wf = source->makeClone(test_particles);
    test_particles.update();
    adapter.installSnapshot(*test_wf, test_particles, parameters);
    const auto reference_log = test_wf->getLogPsi();
    DerivativeValue result{};
    for (std::size_t i = 1; i < points.size(); ++i)
    {
      const auto& point = points[i];
      test_particles.R = initial_positions;
      for (int d = 0; d < 3; ++d) test_particles.R[point.electron][d] = point.replacement_position[d];
      test_particles.update();
      test_wf->evaluateLog(test_particles);
      result += point.bare_weight * std::exp(test_wf->getLogPsi() - reference_log);
    }
    return result;
  };
  for (std::size_t p = 0; p < snapshot->values.size(); ++p)
  {
    auto plus = *snapshot, minus = *snapshot;
    plus.values[p] += step;
    minus.values[p] -= step;
    const auto fd = (nonzero_energy(plus) - nonzero_energy(minus)) / (2 * step);
    CHECK(zero_response.result()[0][p].real() == Catch::Approx(fd.real()).epsilon(3e-7).margin(3e-8));
  }

  // Failed and incomplete captures never bind ECP coverage or publish parameters.
  auto invalid = adapter.makeNonLocalECPDerivativeConsumer(replay, snapshot, 4, 8);
  auto incomplete_op = adapter.makeDerivativeOperator(replay, snapshot, 4, 8);
  NonLocalECPDerivativeContext context{adapter.parameterSchema().providerId(), adapter.parameterSchema().fingerprint(),
      snapshot->version, 4, 8, 1, 1, 123};
  const VJPCoefficientChannel bad_channel{"ecp", DerivativeProduct::LOCAL_ENERGY_VJP,
                                         coefficientsFor(*unbound, unit), ecp_mask};
  context.localization = NonLocalECPLocalization::DLA;
  CHECK_THROWS(invalid->preflight(context, {&bad_channel, 1}, idle_sink));
  context.localization = NonLocalECPLocalization::ORDINARY_LOCALITY;
  invalid->begin(context, {&bad_channel, 1}, idle_sink);
  CHECK_THROWS(invalid->end());
  CHECK_THROWS(invalid->releaseJVPReplay());
  CHECK_FALSE(incomplete_op->capabilities().coversLocalEnergyTerms(ecp_mask));
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

  // Live publication and cold checkpoint restoration must carry the complete
  // physical orbital frame, even when identical active theta has another history.
  source->storeXMLNode(definition.getRoot());
  TrainingWaveFunctionBinding binding(*source, particles, adapterOptions(), *OHMMS::Controller);
  QMCHamiltonian hamiltonian;
  hamiltonian.addOperator(std::make_unique<BareKineticEnergy>(particles), "Kinetic");
  binding.setHamiltonian(hamiltonian);
  ParticleSet live_particles(particles);
  auto live_clone = source->makeClone(live_particles);
  live_particles.update();
  binding.attachLive(*live_clone, live_particles);
  auto& clone_rotation = rotation(*live_clone);
  auto* clone_phi = clone_rotation.Phi_.get();
  const auto positions = particles.R;
  const auto initial_live = binding.snapshotParameters();
  auto invalid_live = initial_live;
  if (OHMMS::Controller->rank() == 0)
    invalid_live.values[0] = std::numeric_limits<double>::quiet_NaN();
  auto* previous_bank = live_lcao.C.get();
  CHECK_THROWS(binding.publishParameters(invalid_live, initial_live.version));
  CHECK(binding.snapshotParameters().values == initial_live.values);
  CHECK(live_lcao.C.get() == previous_bank);
  for (int step = 0; step < 2; ++step)
  {
    auto next = binding.snapshotParameters();
    next.values[step] += step == 0 ? 0.025 : -0.035;
    next.values[2] += 0.0002;
    CHECK(binding.publishParameters(next, next.version) == next.version + 1);
    CHECK(live_rotation.Phi_.get() == fixed_phi);
    CHECK(clone_rotation.Phi_.get() == clone_phi);
    CHECK(live_lcao.C == lcao(clone_rotation).C);
    CHECK(live_rotation.fullRotationCoordinates() == clone_rotation.fullRotationCoordinates());
    source->evaluateLog(particles);
    live_clone->evaluateLog(live_particles);
    CHECK(source->getLogPsi() == live_clone->getLogPsi());
    checkClose(kineticEnergy(particles), kineticEnergy(live_particles));
  }
  const auto saved = binding.snapshotParameters();
  const auto saved_full = live_rotation.fullRotationCoordinates();
  const LCAOrbitalSet::OffloadValueMatrix saved_coefficients(*live_lcao.C);
  const auto checkpoint = std::filesystem::temp_directory_path() /
      ("legacy_orbital_frame_" + std::to_string(OHMMS::Controller->rank()) + "_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".h5");
  FileCleanup cleanup(checkpoint);
  TrainingIterationState coordinator{2, saved.version, binding.parameterSchema().fingerprint()};
  TrainingStageState stage{"orbital", 0, 2, binding.modelIdentity()};
  TrainingCheckpoint::saveAtomic(checkpoint, binding, coordinator, stage);

  // Standalone artifacts must recover the same noncommuting physical frame,
  // not merely its active vector, at identical coordinates without sampler state.
  const auto model = std::filesystem::path(checkpoint.string() + ".parameters.h5");
  FileCleanup model_cleanup(model);
  binding.exportModel(model);
  ParticleSet imported_particles(sibling_particles);
  auto imported = make_private(*sibling, imported_particles, accepted);
  imported->storeXMLNode(definition.getRoot());
  TrainingWaveFunctionBinding imported_binding(*imported, imported_particles, adapterOptions(), *OHMMS::Controller);
  QMCHamiltonian imported_hamiltonian;
  imported_hamiltonian.addOperator(std::make_unique<BareKineticEnergy>(imported_particles), "Kinetic");
  imported_binding.setHamiltonian(imported_hamiltonian);
  auto& imported_rotation = rotation(*imported);
  auto* imported_phi = imported_rotation.Phi_.get();
  imported_binding.importModel(model);
  CHECK(imported_binding.snapshotParameters().values == saved.values);
  CHECK(imported_binding.modelIdentity() == binding.modelIdentity());
  CHECK(imported_rotation.Phi_.get() == imported_phi);
  CHECK(imported_rotation.fullRotationCoordinates() == saved_full);
  for (std::size_t i = 0; i < saved_coefficients.size(); ++i)
    CHECK(lcao(imported_rotation).C->data()[i] == saved_coefficients.data()[i]);
  imported->evaluateLog(imported_particles);
  CHECK(imported->getLogPsi() == source->getLogPsi());
  checkClose(kineticEnergy(imported_particles), kineticEnergy(particles));
  for (std::size_t electron = 0; electron < particles.R.size(); ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      CHECK(imported_particles.R[electron][dimension] == particles.R[electron][dimension]);
  const auto imported_before_failure = imported_binding.snapshotParameters();
  const auto* imported_bank = lcao(imported_rotation).C.get();
  {
    hdf_archive archive;
    REQUIRE(archive.open(model, H5F_ACC_RDWR));
    REQUIRE(H5Ldelete(archive.getFileID(), "/conventional_wavefunction/orbital_frame/spo_0/full_rotation",
                     H5P_DEFAULT) >= 0);
  }
  CHECK_THROWS(imported_binding.importModel(model));
  CHECK(imported_binding.snapshotParameters().version == imported_before_failure.version);
  CHECK(imported_binding.snapshotParameters().values == imported_before_failure.values);
  CHECK(lcao(imported_rotation).C.get() == imported_bank);

  ParticleSet cold_particles(sibling_particles);
  auto cold = make_private(*sibling, cold_particles, accepted);
  cold->storeXMLNode(definition.getRoot());
  TrainingWaveFunctionBinding cold_binding(*cold, cold_particles, adapterOptions(), *OHMMS::Controller);
  QMCHamiltonian cold_hamiltonian;
  cold_hamiltonian.addOperator(std::make_unique<BareKineticEnergy>(cold_particles), "Kinetic");
  cold_binding.setHamiltonian(cold_hamiltonian);
  REQUIRE(cold_binding.modelIdentity() == binding.modelIdentity());
  TrainingIterationState cold_coordinator;
  TrainingStageState cold_stage{"orbital", 0, 0, stage.configuration_fingerprint};
  TrainingCheckpoint::restore(checkpoint, cold_binding, cold_coordinator, cold_stage);
  CHECK(cold_binding.snapshotParameters().values == saved.values);
  CHECK(cold_coordinator.completed_iterations == 2);
  CHECK(cold_stage.completed_stage_iterations == 2);
  auto& cold_rotation = rotation(*cold);
  auto& cold_lcao = lcao(cold_rotation);
  CHECK(cold_rotation.fullRotationCoordinates() == saved_full);
  for (std::size_t i = 0; i < saved_coefficients.size(); ++i)
    CHECK(cold_lcao.C->data()[i] == saved_coefficients.data()[i]);
  cold->evaluateLog(cold_particles);
  CHECK(cold->getLogPsi() == source->getLogPsi());
  checkClose(kineticEnergy(cold_particles), kineticEnergy(particles));
  const auto before_failure = cold_binding.snapshotParameters();
  const auto* before_failure_bank = cold_lcao.C.get();
  {
    hdf_archive archive;
    REQUIRE(archive.open(checkpoint, H5F_ACC_RDWR));
    archive.push("wftrain_checkpoint", false);
    archive.push("orbital_frame", false);
    archive.push("spo_0", false);
    std::vector<double> coefficients;
    REQUIRE(archive.readEntry(coefficients, "accepted_coefficients"));
    coefficients[0] += 0.1;
    archive.write(coefficients, "accepted_coefficients");
  }
  CHECK_THROWS(TrainingCheckpoint::prepareRestore(checkpoint, cold_binding, cold_coordinator, cold_stage));
  CHECK(cold_binding.snapshotParameters().version == before_failure.version);
  CHECK(cold_lcao.C.get() == before_failure_bank);
  TrainingCheckpoint::saveAtomic(checkpoint, binding, coordinator, stage);
  {
    hdf_archive archive;
    REQUIRE(archive.open(checkpoint, H5F_ACC_RDWR));
    REQUIRE(H5Ldelete(archive.getFileID(), "/wftrain_checkpoint/orbital_frame/spo_0/full_rotation", H5P_DEFAULT) >= 0);
  }
  CHECK_THROWS(TrainingCheckpoint::prepareRestore(checkpoint, cold_binding, cold_coordinator, cold_stage));
  CHECK(cold_binding.snapshotParameters().values == before_failure.values);
  CHECK(cold_binding.snapshotParameters().version == before_failure.version);
  CHECK(cold_lcao.C.get() == before_failure_bank);
  for (int electron = 0; electron < particles.getTotalNum(); ++electron)
    for (int dimension = 0; dimension < OHMMS_DIM; ++dimension)
      CHECK(particles.R[electron][dimension] == positions[electron][dimension]);
}

TEST_CASE("Legacy distinct-spin MD orbital products include CI Jastrow and full kinetic tangents",
          "[drivers][training][legacy_adapter][md_orbital]")
{
#ifdef QMC_COMPLEX
  SKIP("This gate qualifies real CPU distinct-owner MD orbital coordinates");
#endif
  ParticleSetPool pool(OHMMS::Controller);
  auto ion_owner = std::make_unique<ParticleSet>(pool.getSimulationCell());
  auto electron_owner = std::make_unique<ParticleSet>(pool.getSimulationCell());
  auto& ions = *ion_owner;
  auto& particles = *electron_owner;
  ions.setName("ion0");
  ions.create({1});
  ions.getSpeciesSet().addSpecies("He");
  ions.R[0] = {0, 0, 0};
  ions.resetGroups();
  ions.update();
  particles.setName("e");
  particles.create({1, 1});
  particles.R[0] = {0.7, 0.2, 0.4};
  particles.R[1] = {-0.5, 0.3, 0.8};
  auto& species = particles.getSpeciesSet();
  const int u = species.addSpecies("u"), d = species.addSpecies("d");
  const int mass = species.addAttribute("mass");
  species(mass, u) = species(mass, d) = 1;
  particles.resetGroups();
  pool.addParticleSet(std::move(ion_owner));
  pool.addParticleSet(std::move(electron_owner));
  Libxml2Document definition;
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
        <rotated_sposet name="rot_up"><sposet name="base_up" size="3">
          <coefficient size="3" id="up_identity">1 0 0 0 1 0 0 0 1</coefficient>
        </sposet></rotated_sposet>
        <rotated_sposet name="rot_dn"><sposet name="base_dn" size="3">
          <coefficient size="3" id="dn_identity">1 0 0 0 1 0 0 0 1</coefficient>
        </sposet></rotated_sposet>
      </sposet_collection>
      <determinantset>
        <multideterminant optimize="yes" spo_0="rot_up" spo_1="rot_dn" algorithm="precomputed_table_method">
          <detlist size="2" type="DETS" nc0="0" ne0="1" nc1="0" ne1="1" nstates="3" cutoff="1e-20">
            <ci id="CI0" coeff="1" occ0="100" occ1="100"/>
            <ci id="CI1" coeff="0.1" occ0="010" occ1="010"/>
          </detlist>
        </multideterminant>
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
  source->storeXMLNode(definition.getRoot());
  particles.update();
  source->evaluateLog(particles);
  auto& md = source->findMSD().front().get();
  const auto owner = [](TrialWaveFunction& wf, int spin) -> RotatedSPOs& {
    return dynamic_cast<RotatedSPOs&>(*wf.findMSD().front().get().getDet(spin).getPhi());
  };
  const auto bank = [&](TrialWaveFunction& wf, int spin) -> LCAOrbitalSet& {
    return dynamic_cast<LCAOrbitalSet&>(*owner(wf, spin).Phi_);
  };
  CHECK(&owner(*source, 0) != &owner(*source, 1));
  CHECK(bank(*source, 0).C != bank(*source, 1).C);
  CHECK_THROWS(md.validatePreparedCoefficientState(md));
  const auto independent_down = bank(*source, 1).C;
  bank(*source, 1).C = bank(*source, 0).C;
  CHECK_THROWS_WITH(md.validatePreparedRotatedCoefficientState(md),
                    Catch::Matchers::ContainsSubstring("independent per-spin physical/reference banks"));
  bank(*source, 1).C = independent_down;
  TrainingWaveFunctionBinding binding(*source, particles, adapterOptions(), *OHMMS::Controller);
  QMCHamiltonian hamiltonian;
  hamiltonian.addOperator(std::make_unique<BareKineticEnergy>(particles), "Kinetic");
  binding.setHamiltonian(hamiltonian);
  const auto snapshot = binding.snapshotParameters();
  REQUIRE(snapshot.values.size() == 11);
  const auto source_log = source->getLogPsi();
  const auto source_kinetic = kineticEnergy(particles);
  const auto source_ci = md.get_C();
  const auto* fixed_map = &md.get_C2node();
  ParticleSet private_particles(particles), sibling_particles(particles);
  auto evaluator = binding.makePrivateClone(private_particles);
  auto sibling = source->makeClone(sibling_particles);
  private_particles.update();
  sibling_particles.update();
  sibling->evaluateLog(sibling_particles);
  const auto sibling_log = sibling->getLogPsi();
  CHECK(bank(*evaluator, 0).C != bank(*source, 0).C);
  CHECK(bank(*evaluator, 1).C != bank(*source, 1).C);
  binding.attachLive(*sibling, sibling_particles);
  auto& adapter = binding.adapter();
  auto op = adapter.makeDerivativeOperator(RefVector<TrialWaveFunction>{*evaluator},
                                            RefVector<ParticleSet>{private_particles}, 0, 0);
  const std::vector<DerivativeValue> coefficient{{0.7, -0.2}};
  const std::array<VJPCoefficientChannel, 2> channels{{
      {"score", DerivativeProduct::SCORE_VJP, coefficientsFor(*op, coefficient), 0},
      {"kinetic", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(*op, coefficient),
       localEnergyTermBit(LocalEnergyTerm::KINETIC)}}};
  RecordingParameterSink transpose, hermitian;
  op->applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::TRANSPOSE, transpose);
  op->applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::HERMITIAN, hermitian);
  std::vector<DerivativeValue> direction(11);
  for (std::size_t p = 0; p < direction.size(); ++p)
    direction[p] = 0.03 * (static_cast<int>(p % 5) - 2);
  StructuredParameterVectorConstView vector(adapter.parameterSchema(), snapshot.version,
                                           {direction.data(), direction.size()});
  RecordingSampleSink score_jvp, energy_jvp;
  op->applyScoreJVP(vector, score_jvp);
  op->applyLocalEnergyJVP(vector, localEnergyTermBit(LocalEnergyTerm::KINETIC), energy_jvp);
  for (std::size_t channel = 0; channel < channels.size(); ++channel)
  {
    DerivativeValue contracted{};
    for (std::size_t p = 0; p < direction.size(); ++p)
      contracted += transpose.result()[channel][p] * direction[p];
    checkClose(contracted, coefficient[0] * (channel == 0 ? score_jvp.result()[0] : energy_jvp.result()[0]));
  }
  const auto probe = [&](const StructuredParameterSnapshot& point) {
    ParticleSet probe_particles(particles);
    auto probe_wf = binding.makePrivateClone(probe_particles);
    probe_particles.update();
    adapter.installSnapshot(*probe_wf, probe_particles, point);
    return std::array<DerivativeValue, 2>{DerivativeValue{probe_wf->getLogPsi()}, kineticEnergy(probe_particles)};
  };
  for (std::size_t p = 0; p < snapshot.values.size(); ++p)
  {
    CAPTURE(p);
    constexpr double epsilon = 1e-5;
    auto plus = snapshot, minus = snapshot;
    plus.values[p] += epsilon;
    minus.values[p] -= epsilon;
    const auto positive = probe(plus), negative = probe(minus);
    for (std::size_t channel = 0; channel < channels.size(); ++channel)
    {
      const auto fd = (positive[channel] - negative[channel]) / (2 * epsilon);
      const auto actual = transpose.result()[channel][p] / coefficient[0];
      CHECK(actual.real() == Catch::Approx(fd.real()).epsilon(2e-6).margin(2e-8));
      CHECK(actual.imag() == Catch::Approx(fd.imag()).margin(2e-8));
      checkClose(hermitian.result()[channel][p], std::conj(actual) * coefficient[0]);
    }
  }
  source->evaluateLog(particles);
  sibling->evaluateLog(sibling_particles);
  CHECK(source->getLogPsi() == source_log);
  CHECK(sibling->getLogPsi() == sibling_log);
  CHECK(kineticEnergy(particles) == source_kinetic);
  CHECK(md.get_C() == source_ci);
  CHECK(&md.get_C2node() == fixed_map);
  CHECK(binding.snapshotParameters().values == snapshot.values);

  // Two independent, noncommuting rotation histories must publish alongside
  // CI/J2 without replacing determinant/SPO identities or either reference C.
  std::array<SPOSet*, 2> fixed_owners{&owner(*source, 0), &owner(*source, 1)};
  std::array<std::vector<QMCTraits::ValueType>, 2> references;
  for (int spin = 0; spin < 2; ++spin)
  {
    const auto& original = bank(*source, spin).originalRotationCoefficients();
    references[spin].assign(original.data(), original.data() + original.size());
  }
  const auto set_delta = [&](StructuredParameterSnapshot& point, const std::string& name, double delta) {
    const auto& variables = adapter.registrationVariables();
    for (std::size_t i = 0; i < variables.size(); ++i)
      if (variables.name(i) == name && variables.where(i) >= 0)
      {
        point.values[variables.where(i)] += delta;
        return;
      }
    FAIL("Required MD fixture coordinate missing: " + name);
  };
  for (int step = 0; step < 2; ++step)
  {
    auto next = binding.snapshotParameters();
    set_delta(next, step == 0 ? "rot_up_orb_rot_0000_0001_r" : "rot_up_orb_rot_0000_0002_r",
              step == 0 ? 0.025 : -0.035);
    set_delta(next, step == 0 ? "rot_dn_orb_rot_0000_0002_r" : "rot_dn_orb_rot_0000_0001_r",
              step == 0 ? -0.017 : 0.021);
    set_delta(next, "CI1", 0.002);
    set_delta(next, "ud_0", 0.0001);
    REQUIRE(binding.publishParameters(next, next.version) == next.version + 1);
    CHECK(md.get_C()[0] == source_ci[0]);
    CHECK(&md.get_C2node() == fixed_map);
    for (int spin = 0; spin < 2; ++spin)
    {
      CHECK(&owner(*source, spin) == fixed_owners[spin]);
      CHECK(bank(*source, spin).C == bank(*sibling, spin).C);
      CHECK(owner(*source, spin).fullRotationCoordinates() == owner(*sibling, spin).fullRotationCoordinates());
      const auto& original = bank(*source, spin).originalRotationCoefficients();
      for (std::size_t i = 0; i < original.size(); ++i)
        CHECK(original.data()[i] == references[spin][i]);
    }
    CHECK(bank(*source, 0).C != bank(*source, 1).C);
    source->evaluateLog(particles);
    sibling->evaluateLog(sibling_particles);
    CHECK(source->getLogPsi() == Catch::Approx(sibling->getLogPsi()).margin(1e-12));
    checkClose(kineticEnergy(particles), kineticEnergy(sibling_particles));
  }
  const auto accepted = binding.snapshotParameters();
  std::array<std::vector<QMCTraits::ValueType>, 2> accepted_banks, accepted_full;
  for (int spin = 0; spin < 2; ++spin)
  {
    const auto& coefficients = *bank(*source, spin).C;
    accepted_banks[spin].assign(coefficients.data(), coefficients.data() + coefficients.size());
    accepted_full[spin] = owner(*source, spin).fullRotationCoordinates();
    REQUIRE(accepted_full[spin].size() == 3);
    CHECK(accepted_full[spin][2] != makeValue(0));
  }
  const auto checkpoint = std::filesystem::temp_directory_path() /
      ("legacy_md_orbital_" + std::to_string(OHMMS::Controller->rank()) + "_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".h5");
  const auto model = std::filesystem::path(checkpoint.string() + ".parameters.h5");
  FileCleanup checkpoint_cleanup(checkpoint), model_cleanup(model);
  TrainingIterationState coordinator{2, accepted.version, binding.parameterSchema().fingerprint()};
  TrainingStageState stage{"md_orbital", 0, 2, binding.modelIdentity()};
  TrainingCheckpoint::saveAtomic(checkpoint, binding, coordinator, stage);
  binding.exportModel(model);
  {
    hdf_archive archive;
    REQUIRE(archive.open(model, H5F_ACC_RDONLY));
    archive.push("conventional_wavefunction", false);
    std::string cookie;
    REQUIRE(archive.readEntry(cookie, "completion_cookie"));
    CHECK(cookie == "conventional-rotated-lcao-mdj-v1");
    archive.push("orbital_frame", false);
    std::vector<int> aliases;
    REQUIRE(archive.readEntry(aliases, "determinant_aliases"));
    CHECK(aliases == std::vector<int>{0, 1});
  }
  ParticleSet cold_particles(particles);
  WaveFunctionFactory cold_factory(cold_particles, pool.getPool(), OHMMS::Controller);
  auto cold = cold_factory.buildTWF(definition.getRoot(), RuntimeOptions{});
  REQUIRE(cold);
  cold->storeXMLNode(definition.getRoot());
  cold_particles.update();
  TrainingWaveFunctionBinding cold_binding(*cold, cold_particles, adapterOptions(), *OHMMS::Controller);
  QMCHamiltonian cold_hamiltonian;
  cold_hamiltonian.addOperator(std::make_unique<BareKineticEnergy>(cold_particles), "Kinetic");
  cold_binding.setHamiltonian(cold_hamiltonian);
  REQUIRE(cold_binding.modelIdentity() == binding.modelIdentity());
  const auto check_frame = [&]() {
    CHECK(cold_binding.snapshotParameters().values == accepted.values);
    for (int spin = 0; spin < 2; ++spin)
    {
      CHECK(owner(*cold, spin).fullRotationCoordinates() == accepted_full[spin]);
      for (std::size_t i = 0; i < accepted_banks[spin].size(); ++i)
        CHECK(bank(*cold, spin).C->data()[i] == accepted_banks[spin][i]);
    }
    cold->evaluateLog(cold_particles);
    CHECK(cold->getLogPsi() == Catch::Approx(source->getLogPsi()).margin(1e-12));
    checkClose(kineticEnergy(cold_particles), kineticEnergy(particles));
  };
  cold_binding.importModel(model);
  check_frame();
  TrainingIterationState restored_coordinator;
  TrainingStageState restored_stage{"md_orbital", 0, 0, stage.configuration_fingerprint};
  TrainingCheckpoint::restore(checkpoint, cold_binding, restored_coordinator, restored_stage);
  check_frame();
  CHECK(restored_coordinator.completed_iterations == 2);
  const auto before_failure = cold_binding.snapshotParameters();
  const auto* before_up = bank(*cold, 0).C.get();
  const auto* before_down = bank(*cold, 1).C.get();
  {
    hdf_archive archive;
    REQUIRE(archive.open(model, H5F_ACC_RDWR));
    REQUIRE(H5Ldelete(archive.getFileID(), "/conventional_wavefunction/orbital_frame/spo_1/full_rotation",
                     H5P_DEFAULT) >= 0);
  }
  CHECK_THROWS(cold_binding.importModel(model));
  CHECK(cold_binding.snapshotParameters().values == before_failure.values);
  CHECK(cold_binding.snapshotParameters().version == before_failure.version);
  CHECK(bank(*cold, 0).C.get() == before_up);
  CHECK(bank(*cold, 1).C.get() == before_down);
  check_frame();

  // One-active-spin input already APP_ABORTs in SlaterDetBuilder; exercise that
  // early input rejection in a native subprocess, never inside this unit test.
  // A constructible BOTH-fixed model checks the validator's source/prepared
  // participant contract without weakening the existing builder guard.
  Libxml2Document fixed_definition;
  fixed_definition.newDoc("test");
  auto* fixed_node = xmlDocCopyNode(definition.getRoot(), fixed_definition.getDocument(), 1);
  REQUIRE(fixed_node);
  fixed_definition.addChild(fixed_node);
  auto* collection = xmlFirstElementChild(fixed_node);
  for (auto* wrapper = xmlFirstElementChild(collection); wrapper;)
  {
    auto* next = xmlNextElementSibling(wrapper);
    if (std::string(reinterpret_cast<const char*>(wrapper->name)) == "rotated_sposet")
    {
      auto* spo = xmlFirstElementChild(wrapper);
      auto* name = xmlGetProp(wrapper, BAD_CAST "name");
      REQUIRE(name);
      xmlUnlinkNode(spo);
      xmlSetProp(spo, BAD_CAST "name", name);
      xmlFree(name);
      xmlReplaceNode(wrapper, spo);
      xmlFreeNode(wrapper);
    }
    wrapper = next;
  }
  WaveFunctionFactory fixed_factory(particles, pool.getPool(), OHMMS::Controller);
  auto fixed_wf = fixed_factory.buildTWF(fixed_node, RuntimeOptions{});
  REQUIRE(fixed_wf);
  auto& fixed_md = fixed_wf->findMSD().front().get();
  CHECK_THROWS_WITH(md.validatePreparedRotatedCoefficientState(fixed_md),
                    Catch::Matchers::ContainsSubstring("two distinct active spin owners"));
  CHECK_THROWS_WITH(fixed_md.validatePreparedRotatedCoefficientState(md),
                    Catch::Matchers::ContainsSubstring("two distinct active spin owners"));
  CHECK(binding.snapshotParameters().values == accepted.values);
}

TEST_CASE("Legacy private J1 and J2 resets preserve live physical banks on rejection",
          "[drivers][training][legacy_adapter][jastrow][private_banks]")
{
  const SimulationCell cell;
  auto particles = makeTwoSpeciesParticles(cell);
  ParticleSet ions(cell);
  ions.setName("ion0");
  ions.create({1});
  ions.R[0] = {0, 0, 0};
  ions.getSpeciesSet().addSpecies("H");
  ions.resetGroups();
  ions.update();
  auto source = makeBsplineJ2WaveFunction(particles, {0.12, -0.08, 0.04, -0.01}, 0.37);
  using Functor = BsplineFunctor<QMCTraits::RealType>;
  auto functor = std::make_unique<Functor>("private_j1_functor");
  functor->cutoff_radius = 4;
  functor->resize(4);
  for (int i = 0; i < 4; ++i)
  {
    functor->Parameters[i] = 0.03 * (i + 1);
    functor->myVars.insert("private_j1_" + std::to_string(i), functor->Parameters[i], true, optimize::LOGLINEAR_P);
  }
  functor->reset();
  auto j1 = std::make_unique<J1OrbitalSoA<Functor>>("private_j1", ions, particles, false);
  j1->addFunc(0, std::move(functor));
  source->addComponent(std::move(j1));
  particles.update();
  source->evaluateLog(particles);
  const auto source_log = source->getLogPsi();
  const auto source_kinetic = kineticEnergy(particles);
  ParticleSet sibling_particles(particles), private_particles(particles);
  auto sibling = source->makeClone(sibling_particles);
  auto evaluator = source->makeClone(private_particles);
  sibling_particles.update();
  private_particles.update();
  sibling->evaluateLog(sibling_particles);
  // Independently evaluated clones can differ by an initial reduction-roundoff
  // bit. Isolation must preserve each object's OWN pre-reset physical baseline.
  const auto sibling_log = sibling->getLogPsi();
  const auto sibling_kinetic = kineticEnergy(sibling_particles);
  CAPTURE(source_log, sibling_log);
  source->evaluateLog(particles);
  sibling->evaluateLog(sibling_particles);
  CHECK(source->getLogPsi() == source_log);
  CHECK(sibling->getLogPsi() == sibling_log);
  CHECK(kineticEnergy(particles) == source_kinetic);
  CHECK(kineticEnergy(sibling_particles) == sibling_kinetic);
  const auto banks = [](TrialWaveFunction& wf) {
    std::vector<const void*> result;
    for (OptimizableObject& object : wf.extractOptimizableObjectRefs())
      if (auto* spline = dynamic_cast<Functor*>(&object))
        result.push_back(spline->spline_coefs_.get());
    return result;
  };
  REQUIRE(banks(*source).size() == 2);
  CHECK(banks(*source) == banks(*sibling));
  CHECK(banks(*source) == banks(*evaluator));
  TrainingWaveFunctionBinding binding(*source, particles, adapterOptions(), *OHMMS::Controller);
  QMCHamiltonian hamiltonian;
  hamiltonian.addOperator(std::make_unique<BareKineticEnergy>(particles), "Kinetic");
  binding.setHamiltonian(hamiltonian);
  binding.attachLive(*sibling, sibling_particles);
  const auto snapshot = binding.snapshotParameters();
  REQUIRE(snapshot.values.size() == 8);
  auto displaced = snapshot;
  for (auto& value : displaced.values)
    value += 0.03;
  binding.adapter().installSnapshot(*evaluator, private_particles, displaced);
  CHECK(banks(*evaluator) != banks(*source));
  const auto private_banks = banks(*evaluator);
  binding.adapter().installSnapshot(*evaluator, private_particles, displaced);
  CHECK(banks(*evaluator) == private_banks);
  CHECK(evaluator->getLogPsi() != source_log);
  const auto check_physical_source = [&]() {
    source->evaluateLog(particles);
    sibling->evaluateLog(sibling_particles);
    CHECK(source->getLogPsi() == source_log);
    CHECK(sibling->getLogPsi() == sibling_log);
    CHECK(kineticEnergy(particles) == source_kinetic);
    CHECK(kineticEnergy(sibling_particles) == sibling_kinetic);
    CHECK(binding.snapshotParameters().values == snapshot.values);
    CHECK(binding.snapshotParameters().version == snapshot.version);
  };
  // Do NOT restore the private point before checking physical source isolation.
  check_physical_source();
  auto rejected = snapshot;
  std::fill(rejected.values.begin(), rejected.values.end(), 1e200);
  CHECK_THROWS_WITH(binding.publishParameters(rejected, snapshot.version),
                    Catch::Matchers::ContainsSubstring("Nonfinite conventional candidate energy"));
  check_physical_source();
  CHECK(banks(*source) == banks(*sibling));
}

TEST_CASE("Legacy Gamma HEG products preserve scalar domains, periodicity and full Ewald derivatives",
          "[drivers][training][legacy_adapter][periodic_heg]")
{
  // Isolate test-global breakup selection without changing production singleton
  // semantics. Existing operators retain their own immutable built handlers.
  struct RestoreBreakup
  {
    decltype(LRCoulombSingleton::CoulombHandler) energy = std::move(LRCoulombSingleton::CoulombHandler);
    decltype(LRCoulombSingleton::CoulombDerivHandler) derivative = std::move(LRCoulombSingleton::CoulombDerivHandler);
    LRCoulombSingleton::lr_type kind = LRCoulombSingleton::this_lr_type;
    ~RestoreBreakup()
    {
      LRCoulombSingleton::CoulombHandler = std::move(energy);
      LRCoulombSingleton::CoulombDerivHandler = std::move(derivative);
      LRCoulombSingleton::this_lr_type = kind;
    }
  } restore_breakup;
  struct HEG
  {
    ParticleSetPool pool{OHMMS::Controller};
    Libxml2Document cell, definition;
    ParticleSet* particles;
    std::unique_ptr<TrialWaveFunction> wavefunction;
    QMCHamiltonian hamiltonian;
    CoulombPBCAA* coulomb;
    HEG(const std::string& rs, const std::string& cutoff)
    {
      LRCoulombSingleton::CoulombHandler.reset();
      LRCoulombSingleton::CoulombDerivHandler.reset();
      const std::string xml = "<simulationcell><parameter name='rs' condition='14'>" + rs +
          "</parameter><parameter name='bconds'>p p p</parameter><parameter name='LR_dim_cutoff'>" + cutoff +
          "</parameter><parameter name='LR_handler'>ewald</parameter></simulationcell>";
      REQUIRE(cell.parseFromString(xml));
      REQUIRE(pool.readSimulationCellXML(cell.getRoot()));
      auto owner = std::make_unique<ParticleSet>(pool.getSimulationCell());
      particles = owner.get();
      particles->setName("e");
      particles->create({7, 7});
      auto& species = particles->getSpeciesSet();
      const int u = species.addSpecies("u"), d = species.addSpecies("d");
      const int charge = species.addAttribute("charge"), mass = species.addAttribute("mass");
      species(charge, u) = species(charge, d) = -1;
      species(mass, u) = species(mass, d) = 1;
      particles->resetGroups();
      const double length = particles->getLattice().R(0, 0);
      for (int i = 0; i < 14; ++i)
        particles->R[i] = {length * std::fmod(0.137 + i * 0.217 + i * i * 0.037, 1.0),
                           length * std::fmod(0.231 + i * 0.371 + i * i * 0.017, 1.0),
                           length * std::fmod(0.317 + i * 0.127 + i * i * 0.053, 1.0)};
      // Exercise a close pair straddling a periodic cell face.
      particles->R[0] = {0.01 * length, 0.21 * length, 0.31 * length};
      particles->R[7] = {0.99 * length, 0.22 * length, 0.33 * length};
      particles->createSK();
      pool.addParticleSet(std::move(owner));
      // Existing heg_14_gamma/heg.sj.wfs.xml physical model and coefficients.
      REQUIRE(definition.parseFromString(R"(
        <wavefunction name="psi0" target="e">
          <sposet_collection type="free"><sposet name="spo-ud" size="7" optimize="no"/></sposet_collection>
          <determinantset><slaterdeterminant>
            <determinant sposet="spo-ud"/><determinant sposet="spo-ud"/>
          </slaterdeterminant></determinantset>
          <jastrow name="J2" type="Two-Body" function="Bspline" optimize="yes">
            <correlation speciesA="u" speciesB="u" size="5">
              <coefficients id="uu" type="Array" optimize="yes">1.082858193 0.6653279375 0.4358910287 0.2243616172 0.1102948764</coefficients>
            </correlation>
            <correlation speciesA="u" speciesB="d" size="5">
              <coefficients id="ud" type="Array" optimize="yes">1.696171854 1.047722154 0.6275148566 0.3175982878 0.1446706214</coefficients>
            </correlation>
          </jastrow>
        </wavefunction>)"));
      WaveFunctionFactory factory(*particles, pool.getPool(), OHMMS::Controller);
      wavefunction = factory.buildTWF(definition.getRoot(), RuntimeOptions{});
      REQUIRE(wavefunction);
      wavefunction->storeXMLNode(definition.getRoot());
      hamiltonian.addOperator(std::make_unique<BareKineticEnergy>(*particles), "Kinetic");
      auto ewald = std::make_unique<CoulombPBCAA>(*particles, true, false, false);
      coulomb = ewald.get();
      hamiltonian.addOperator(std::move(ewald), "ElecElec");
      hamiltonian.addObservables(*particles);
      particles->update();
      wavefunction->evaluateLog(*particles);
    }
  } model("5", "6");
  auto& particles = *model.particles;
  auto& source = *model.wavefunction;
#ifndef QMC_COMPLEX
  TrainingWaveFunctionBinding binding(source, particles, adapterOptions(), *OHMMS::Controller);
  binding.setHamiltonian(model.hamiltonian);
  const auto identity = binding.modelIdentity();
  auto& adapter = binding.adapter();
#else
  // Numerical proof precedes complex Binding/driver admission. The active
  // coordinates are real J2 values, not complex orbital/phase parameters.
  LegacyWaveFunctionAdapter adapter("conventional", source, adapterOptions());
#endif
  const auto snapshot = adapter.snapshotParameters();
  REQUIRE(snapshot.values.size() == 10);
  const auto original_positions = particles.R;
  const auto original_log = source.getLogPsi();
  const auto original_kinetic = kineticEnergy(particles);
  const auto original_phase = source.getPhase();
  ParticleSet sibling_particles(particles);
  auto sibling = source.makeClone(sibling_particles);
  sibling_particles.update(); sibling->evaluateLog(sibling_particles);
  const auto sibling_log = sibling->getLogPsi(), sibling_phase = sibling->getPhase();
  const auto sibling_kinetic = kineticEnergy(sibling_particles);
  class CountEwaldRandom final : public FakeRandom<QMCTraits::FullPrecRealType>
  {
  public:
    QMCTraits::FullPrecRealType operator()() override { ++draws; return 0.5; }
    std::size_t draws = 0;
  } rng;
  model.hamiltonian.setRandomGenerator(&rng);
  const auto original_potential = model.coulomb->evaluate(particles);
  const auto definition = model.coulomb->ewaldModelDefinition();
  CHECK(definition.find("handler=EwaldHandler3D;") != std::string::npos);
  CHECK(definition.find("neutralizing-background+self-v1") != std::string::npos);
  LRCoulombSingleton::this_lr_type = LRCoulombSingleton::ESLER;
  CHECK(model.coulomb->ewaldModelDefinition() == definition);

  // A genuine periodic SPO has periodic nonconstant values, gradients AND
  // Laplacians, not merely a periodic pair potential around open orbitals.
  auto& spo = source.findSD().front().get().getPhi();
  SPOSet::ValueVector values(7), laplacians(7), shifted_values(7), shifted_laplacians(7);
  SPOSet::GradVector gradients(7), shifted_gradients(7);
  spo.evaluateVGL(particles, 0, values, gradients, laplacians);
  CHECK(std::abs(values[1] - values[2]) > 1e-4);
#ifdef QMC_COMPLEX
  REQUIRE(std::any_of(values.begin(), values.end(), [](auto value) { return std::abs(value.imag()) > 0.1; }));
#endif
  for (int axis = 0; axis < OHMMS_DIM; ++axis)
  {
    for (int d = 0; d < OHMMS_DIM; ++d)
      particles.R[0][d] += particles.getLattice().R(axis, d);
    particles.update();
    spo.evaluateVGL(particles, 0, shifted_values, shifted_gradients, shifted_laplacians);
    for (int orbital = 0; orbital < 7; ++orbital)
    {
      checkClose(derivativeValue(values[orbital]), derivativeValue(shifted_values[orbital]));
      checkClose(derivativeValue(laplacians[orbital]), derivativeValue(shifted_laplacians[orbital]));
      for (int d = 0; d < OHMMS_DIM; ++d)
        checkClose(derivativeValue(gradients[orbital][d]), derivativeValue(shifted_gradients[orbital][d]));
    }
    source.evaluateLog(particles);
    CHECK(source.getLogPsi() == Catch::Approx(original_log).margin(1e-10));
    checkClose(kineticEnergy(particles), original_kinetic);
    CHECK(model.coulomb->evaluate(particles) == Catch::Approx(original_potential).margin(1e-10));
    particles.R = original_positions;
    particles.update();
  }
  source.evaluateLog(particles);
  ParticleSet private_particles(particles);
  auto evaluator = source.makeClone(private_particles);
  private_particles.update();
  auto op = adapter.makeDerivativeOperator(RefVector<TrialWaveFunction>{*evaluator},
                                            RefVector<ParticleSet>{private_particles}, 0, 0);
  CHECK(op->capabilities().parameter_scalar_domain == ParameterScalarDomain::REAL64);
  CHECK(op->capabilities().result_scalar_domain == ParameterScalarDomain::COMPLEX128);
  for (const auto& block : adapter.parameterSchema().blocks())
    CHECK(block.scalar_domain == ParameterScalarDomain::REAL64);
#ifdef QMC_COMPLEX
  CHECK(op->storageDiagnostics().real_parameter_vectors == 0);
  CHECK(op->storageDiagnostics().complex_parameter_vectors == 5);
#endif
  const std::vector<DerivativeValue> coefficient{{0.7, -0.2}};
  const std::array<VJPCoefficientChannel, 2> channels{{
      {"score", DerivativeProduct::SCORE_VJP, coefficientsFor(*op, coefficient), 0},
      {"total_energy", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(*op, coefficient),
       localEnergyTermBit(LocalEnergyTerm::KINETIC)}}};
  // Ewald has no J2 parameter dependence: its derivative is zero, but the FD
  // oracle below evaluates the complete physical kinetic+Ewald local energy.
  RecordingParameterSink transpose, hermitian;
  op->applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::TRANSPOSE, transpose);
  op->applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::HERMITIAN, hermitian);
  std::vector<DerivativeValue> direction(10);
  for (std::size_t p = 0; p < direction.size(); ++p)
    direction[p] = 0.03 * (static_cast<int>(p % 5) - 2);
  StructuredParameterVectorConstView vector(adapter.parameterSchema(), snapshot.version,
                                           {direction.data(), direction.size()});
  RecordingSampleSink score_jvp, energy_jvp;
  op->applyScoreJVP(vector, score_jvp);
  op->applyLocalEnergyJVP(vector, localEnergyTermBit(LocalEnergyTerm::KINETIC), energy_jvp);
  for (std::size_t channel = 0; channel < channels.size(); ++channel)
  {
    DerivativeValue contraction{};
    for (std::size_t p = 0; p < direction.size(); ++p)
      contraction += transpose.result()[channel][p] * direction[p];
    checkClose(contraction, coefficient[0] * (channel == 0 ? score_jvp.result()[0] : energy_jvp.result()[0]));
  }
  const auto probe = [&](const StructuredParameterSnapshot& point) {
    ParticleSet probe_particles(particles);
    auto probe_wavefunction = source.makeClone(probe_particles);
    probe_particles.update();
    adapter.installSnapshot(*probe_wavefunction, probe_particles, point);
    return std::array<DerivativeValue, 2>{DerivativeValue{probe_wavefunction->getLogPsi()},
        kineticEnergy(probe_particles) + DerivativeValue{model.coulomb->evaluate(probe_particles)}};
  };
  for (std::size_t p = 0; p < snapshot.values.size(); ++p)
  {
    CAPTURE(p);
    constexpr double epsilon = 1e-5;
    auto plus = snapshot, minus = snapshot;
    plus.values[p] += epsilon;
    minus.values[p] -= epsilon;
    const auto positive = probe(plus), negative = probe(minus);
    for (std::size_t channel = 0; channel < channels.size(); ++channel)
    {
      const auto fd = (positive[channel] - negative[channel]) / (2 * epsilon);
      const auto actual = transpose.result()[channel][p] / coefficient[0];
      CHECK(actual.real() == Catch::Approx(fd.real()).epsilon(2e-6).margin(2e-8));
      CHECK(actual.imag() == Catch::Approx(fd.imag()).margin(2e-8));
      checkClose(hermitian.result()[channel][p], std::conj(actual) * coefficient[0]);
    }
  }
  CHECK(adapter.snapshotParameters().values == snapshot.values);
  source.evaluateLog(particles);
  CHECK(source.getLogPsi() == original_log);
  CHECK(source.getPhase() == original_phase);
  checkClose(kineticEnergy(particles), original_kinetic);
  sibling->evaluateLog(sibling_particles);
  CHECK(sibling->getLogPsi() == sibling_log);
  CHECK(sibling->getPhase() == sibling_phase);
  checkClose(kineticEnergy(sibling_particles), sibling_kinetic);
  CHECK(rng.draws == 0);
  for (int electron = 0; electron < particles.getTotalNum(); ++electron)
  {
    CHECK(particles.R[electron] == original_positions[electron]);
    CHECK(sibling_particles.R[electron] == original_positions[electron]);
  }
  CHECK(&source.findSD().front().get().getPhi() == &spo);
  CHECK(op->storageDiagnostics().shared_workspace_bytes <= adapterOptions().maximum_derivative_scratch_bytes);

  HEG changed_cutoff("5", "7"), changed_cell("5.1", "6");
#ifdef QMC_COMPLEX
  TrainingWaveFunctionBinding binding(source, particles, adapterOptions(), *OHMMS::Controller);
  binding.setHamiltonian(model.hamiltonian);
  const auto identity = binding.modelIdentity();
#endif
  TrainingWaveFunctionBinding cutoff_binding(*changed_cutoff.wavefunction, *changed_cutoff.particles,
                                               adapterOptions(), *OHMMS::Controller);
  cutoff_binding.setHamiltonian(changed_cutoff.hamiltonian);
  TrainingWaveFunctionBinding cell_binding(*changed_cell.wavefunction, *changed_cell.particles,
                                             adapterOptions(), *OHMMS::Controller);
  cell_binding.setHamiltonian(changed_cell.hamiltonian);
  CHECK(cutoff_binding.modelIdentity() != identity);
  CHECK(cell_binding.modelIdentity() != identity);
#ifdef QMC_COMPLEX
  // Publication and cold import retain real canonical coordinates, immutable
  // complex SPOs and the built Ewald/scalar-domain scientific identity.
  auto candidate = binding.snapshotParameters();
  candidate.values[0] += 0.001;
  ParticleSet expected_particles(particles);
  expected_particles.update();
  auto expected = binding.makeCandidate(expected_particles, candidate);
  const auto expected_energy = kineticEnergy(expected_particles);
  CHECK(binding.publishParameters(candidate, candidate.version) == candidate.version + 1);
  source.evaluateLog(particles);
  CHECK(source.getLogPsi() == Catch::Approx(expected->getLogPsi()).margin(1e-12));
  checkClose(kineticEnergy(particles), expected_energy);
  CHECK(&source.findSD().front().get().getPhi() == &spo);
  const auto file = std::filesystem::temp_directory_path() /
      ("complex_heg_identity_" + std::to_string(OHMMS::Controller->rank()) + "_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".h5");
  FileCleanup cleanup(file);
  binding.exportModel(file);
  CHECK(binding.readModel(file).values == candidate.values);
  std::string physical;
  {
    hdf_archive input;
    REQUIRE(input.open(file, H5F_ACC_RDONLY));
    input.push("conventional_wavefunction", false);
    REQUIRE(input.readEntry(physical, "physical_definition"));
  }
  const std::string domains = "parameter=REAL64;wavefunction=COMPLEX128;";
  REQUIRE(physical.find(domains) != std::string::npos);
  CHECK(physical.find("complex_Gamma_FreeOrbital_kvectors_v1:") != std::string::npos);
  HEG fresh("5", "6");
  TrainingWaveFunctionBinding imported(*fresh.wavefunction, *fresh.particles, adapterOptions(), *OHMMS::Controller);
  imported.setHamiltonian(fresh.hamiltonian);
  const auto initial = imported.snapshotParameters();
  // A valid artifact is insufficient without successful local profile/Ham
  // admission. A failed admission must not leave a usable partial identity.
  TrainingWaveFunctionBinding unattached(*fresh.wavefunction, *fresh.particles, adapterOptions(), *OHMMS::Controller);
  CHECK_THROWS_WITH(unattached.importModel(file), Catch::Matchers::ContainsSubstring("attached physical Hamiltonian"));
  CHECK(unattached.snapshotParameters().version == 0);
  CHECK(unattached.snapshotParameters().values == initial.values);
  HEG unsupported("5", "6");
  unsupported.particles->setTwist({0.1, 0, 0});
  TrainingWaveFunctionBinding rejected(*unsupported.wavefunction, *unsupported.particles, adapterOptions(), *OHMMS::Controller);
  CHECK_THROWS_WITH(rejected.setHamiltonian(unsupported.hamiltonian), Catch::Matchers::ContainsSubstring("does not admit twist"));
  CHECK_THROWS_WITH(rejected.importModel(file), Catch::Matchers::ContainsSubstring("attached physical Hamiltonian"));
  CHECK(rejected.snapshotParameters().version == 0);
  CHECK(rejected.snapshotParameters().values == initial.values);
  {
    // A real-representation payload cannot masquerade as the complex model,
    // even with the same XML, real parameter layout and completion cookie.
    auto other_domain = physical;
    other_domain.replace(other_domain.find(domains), domains.size(), "parameter=REAL64;wavefunction=REAL64;");
    hdf_archive output;
    REQUIRE(output.open(file, H5F_ACC_RDWR));
    output.push("conventional_wavefunction", false);
    output.write(other_domain, "physical_definition");
  }
  CHECK_THROWS(imported.importModel(file));
  CHECK(imported.snapshotParameters().values == initial.values);
  CHECK(imported.snapshotParameters().version == initial.version);
  fresh.wavefunction->evaluateLog(*fresh.particles);
  CHECK(fresh.wavefunction->getLogPsi() == original_log);
  checkClose(kineticEnergy(*fresh.particles), original_kinetic);
  {
    hdf_archive output;
    REQUIRE(output.open(file, H5F_ACC_RDWR));
    output.push("conventional_wavefunction", false);
    output.write(physical, "physical_definition");
  }
  CHECK_THROWS(cutoff_binding.importModel(file));
  CHECK_THROWS(cell_binding.importModel(file));
  CHECK(cutoff_binding.snapshotParameters().version == 0);
  CHECK(cell_binding.snapshotParameters().version == 0);
  imported.importModel(file);
  fresh.wavefunction->evaluateLog(*fresh.particles);
  CHECK(imported.snapshotParameters().values == candidate.values);
  CHECK(imported.snapshotParameters().version == initial.version + 1);
  CHECK(fresh.wavefunction->getLogPsi() == Catch::Approx(expected->getLogPsi()).margin(1e-12));
  CHECK(fresh.wavefunction->getPhase() == Catch::Approx(expected->getPhase()).margin(1e-12));
  checkClose(kineticEnergy(*fresh.particles), expected_energy);
  CHECK(fresh.coulomb->evaluate(*fresh.particles) == Catch::Approx(original_potential).margin(1e-12));
  CHECK(rng.draws == 0);
#endif
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

TEST_CASE("Accepted-reference virtual determinants remain finite at exact nodes",
          "[legacy_adapter][accepted_reference_virtual]")
{
#ifdef QMC_COMPLEX
  SKIP("Accepted-reference primitive qualifies only fixed real CPU orbitals");
#endif
  class PolynomialSPO final : public SPOSet
  {
  public:
    explicit PolynomialSPO(bool transverse_factor = false)
        : SPOSet("accepted_reference_polynomial", 4), transverse_factor(transverse_factor) {}
    std::string getClassName() const override { return "PolynomialSPO"; }
    std::unique_ptr<SPOSet> makeClone() const override { return std::make_unique<PolynomialSPO>(transverse_factor); }
    void evaluateValue(const ParticleSet& p, int i, ValueVector& values) override
    {
      const double x = p.R[i][0];
      const double factor = transverse_factor ? 1 + p.R[i][1] + p.R[i][2] : 1;
      values[0] = factor; values[1] = factor * x * x; values[2] = factor * x; values[3] = factor * x * x * x;
    }
    void evaluateVGL(const ParticleSet& p, int i, ValueVector& values, GradVector& gradients,
                     ValueVector& laplacians) override
    {
      evaluateValue(p, i, values);
      gradients = GradType{};
      laplacians = ValueType{};
      const double x = p.R[i][0];
      const double factor = transverse_factor ? 1 + p.R[i][1] + p.R[i][2] : 1;
      gradients[1][0] = factor * 2 * x; gradients[2][0] = factor; gradients[3][0] = factor * 3 * x * x;
      // The common factor is linear in y/z and independent of x: its
      // Laplacian and grad(factor).grad(polynomial) product term both vanish.
      laplacians[1] = factor * 2; laplacians[3] = factor * 6 * x;
      if (transverse_factor)
      {
        const double base[] = {1, x * x, x, x * x * x};
        for (int orbital = 0; orbital < 4; ++orbital)
          gradients[orbital][1] = gradients[orbital][2] = base[orbital];
      }
    }
    bool transverse_factor;
    void evaluate_notranspose(const ParticleSet& p, int first, int last, ValueMatrix& values,
                               GradMatrix& gradients, ValueMatrix& laplacians) override
    {
      for (int i = first; i < last; ++i)
      {
        ValueVector v(values[i - first], 4), l(laplacians[i - first], 4);
        GradVector g(gradients[i - first], 4);
        evaluateVGL(p, i, v, g, l);
      }
    }
  };
  const SimulationCell cell;
  ParticleSet accepted(cell), virtual_positions(cell);
  accepted.create({2, 2});
  virtual_positions.create({1});
  for (int i = 0; i < 4; ++i) accepted.R[i] = {double(1 + i % 2), 0, 0};
  accepted.update();
  const auto original_positions = accepted.R;
  std::vector<ci_configuration2> configurations(4);
  configurations[0].occup = {0, 1}; configurations[1].occup = {0, 2};
  configurations[2].occup = {1, 2}; configurations[3].occup = {2, 3};
  // Dense derivative-row oracle: two reference choices exercise zero transverse
  // gradients and zero Laplacians, retaining rank-2 excitations/signs.
  for (int reference : {0, 1})
    for (int spin = 0; spin < 2; ++spin)
    {
      MultiDiracDeterminant block(std::make_unique<PolynomialSPO>(), false, 2 * spin, 2);
      std::vector<std::size_t> map;
      block.createDetData(reference, configurations, {0, 1, 2, 3}, map);
      ParticleSet derivative_particles(accepted);
      const double first_x = reference == 0 ? 0 : 1;
      derivative_particles.R[2 * spin][0] = first_x;
      derivative_particles.update();
      block.evaluateForWalkerMove(derivative_particles);
      const auto values = [](double x, int derivative) {
        if (derivative == 0) return std::array<double, 4>{1, x * x, x, x * x * x};
        if (derivative == 1) return std::array<double, 4>{0, 2 * x, 1, 3 * x * x};
        if (derivative == 2) return std::array<double, 4>{0, 2, 0, 6 * x};
        return std::array<double, 4>{0, 0, 0, 0};
      };
      const auto x = values(first_x, 0), y = values(2, 0);
      const auto& ref = configurations[reference].occup;
      const double denominator = x[ref[0]] * y[ref[1]] - x[ref[1]] * y[ref[0]];
      for (int determinant = 0; determinant < 4; ++determinant)
        for (int row = 0; row < 2; ++row)
        {
          const auto& occ = block.getConfiguration(determinant).occup;
          const auto oracle = [&](int derivative) {
            const auto replacement = values(row == 0 ? first_x : 2, derivative);
            return (row == 0 ? replacement[occ[0]] * y[occ[1]] - replacement[occ[1]] * y[occ[0]]
                             : x[occ[0]] * replacement[occ[1]] - x[occ[1]] * replacement[occ[0]]) / denominator;
          };
          CHECK(std::real(block.getGrads()(determinant, row)[0]) == Catch::Approx(oracle(1)).margin(3e-13));
          for (int dimension = 1; dimension < 3; ++dimension)
            CHECK(std::real(block.getGrads()(determinant, row)[dimension]) == Catch::Approx(oracle(3)).margin(3e-13));
          CHECK(std::real(block.getLapls()(determinant, row)) == Catch::Approx(oracle(2)).margin(3e-13));
        }
      // Also check orbital substitution against the alternate accepted {0,2}
      // reference ordering; keep its original denominator fixed at this base.
      virtual_positions.R[0] = {3, 0, 0};
      Vector<QMCTraits::ValueType> ratios(4), generated(4);
      const auto a = values(3, 0);
      for (const auto [p, q] : {std::pair{0, 3}, std::pair{1, 2}})
      {
        block.evaluateOrbitalGeneratorToAcceptedReference(virtual_positions, 0, 2 * spin, p, q, ratios, generated);
        std::array<double, 4> da{}, db{};
        da[p] = a[q]; da[q] = -a[p]; db[p] = y[q]; db[q] = -y[p];
        for (int term = 0; term < 4; ++term)
        {
          const auto& occ = block.getConfiguration(term).occup;
          const int i = occ[0], j = occ[1];
          CHECK(std::real(ratios[term]) == Catch::Approx((a[i] * y[j] - a[j] * y[i]) / denominator).margin(3e-13));
          CHECK(std::real(generated[term]) == Catch::Approx(
              (da[i] * y[j] + a[i] * db[j] - da[j] * y[i] - a[j] * db[i]) / denominator).margin(3e-13));
        }
      }
    }
  std::array<std::unique_ptr<MultiDiracDeterminant>, 2> spins;
  std::array<std::vector<std::size_t>, 2> mappings;
  for (int spin = 0; spin < 2; ++spin)
  {
    spins[spin] = std::make_unique<MultiDiracDeterminant>(std::make_unique<PolynomialSPO>(), false, 2 * spin, 2);
    spins[spin]->createDetData(0, configurations, {0, 1, 2, 3}, mappings[spin]);
    spins[spin]->evaluateForWalkerMove(accepted);
  }
  // Independent occupied 2x2 determinant, not the rank-(k+1) implementation.
  const auto determinant = [](const ci_configuration2& configuration, double first, double second) {
    const std::array<double, 4> a{1, first * first, first, first * first * first};
    const std::array<double, 4> b{1, second * second, second, second * second * second};
    return a[configuration.occup[0]] * b[configuration.occup[1]] -
        a[configuration.occup[1]] * b[configuration.occup[0]];
  };
  // Independent column substitution: +phi_q into p and -phi_p into q.
  // Its denominator stays the ORIGINAL accepted reference, not a rotated one.
  const auto generated_determinant = [](const ci_configuration2& configuration, double first, double second,
                                       int p, int q, double theta) {
    std::array<double, 4> a{1, first * first, first, first * first * first};
    std::array<double, 4> b{1, second * second, second, second * second * second};
    std::array<double, 4> da{}, db{};
    da[p] = a[q]; da[q] = -a[p]; db[p] = b[q]; db[q] = -b[p];
    const int i = configuration.occup[0], j = configuration.occup[1];
    const double derivative = da[i] * b[j] + a[i] * db[j] - da[j] * b[i] - a[j] * db[i];
    for (int orbital = 0; orbital < 4; ++orbital)
    { a[orbital] += theta * da[orbital]; b[orbital] += theta * db[orbital]; }
    return std::make_pair(a[i] * b[j] - a[j] * b[i], derivative);
  };
  for (int spin = 0; spin < 2; ++spin)
    for (int moved = 0; moved < 2; ++moved)
    {
      CAPTURE(spin, moved);
      auto& block = *spins[spin];
      const int particle = 2 * spin + moved;
      const auto original_log = block.getLogValueRefDet();
      const std::vector<QMCTraits::ValueType> original_ratios(
          block.getRatiosToRefDet().begin(), block.getRatiosToRefDet().end());
      virtual_positions.R[0] = {3, 0, 0};
      block.evaluateDetsForPtclMove(virtual_positions, 0, particle);
      const auto regular_reference = block.getRefDetRatio();
      const std::vector<QMCTraits::ValueType> regular_ratios(
          block.getNewRatiosToRefDet().begin(), block.getNewRatiosToRefDet().end());
      Vector<QMCTraits::ValueType> actual(4);
      Vector<QMCTraits::ValueType> orbital_derivatives(4);
      const auto* output_storage = actual.data();
      const auto* derivative_storage = orbital_derivatives.data();
      for (double replacement : {-2., -1., 0., 1., 2., 3.})
      {
        CAPTURE(replacement);
        virtual_positions.R[0] = {replacement, 0, 0};
        block.evaluateDetsToAcceptedReference(virtual_positions, 0, particle, actual);
        CHECK(actual.data() == output_storage);
        for (int term = 0; term < 4; ++term)
        {
          const auto& configuration = block.getConfiguration(term);
          const double expected = determinant(configuration, moved == 0 ? replacement : 1,
              moved == 1 ? replacement : 2) / 3;
          CHECK(std::real(actual[term]) == Catch::Approx(expected).epsilon(2e-13).margin(2e-13));
          CHECK(block.getRatiosToRefDet()[term] == original_ratios[term]);
        }
        CHECK(block.getLogValueRefDet() == original_log);
        const std::vector<QMCTraits::ValueType> base_ratios(actual.begin(), actual.end());
        for (const auto [p, q] : {std::pair{0, 1}, std::pair{0, 2}, std::pair{0, 3}, std::pair{1, 2}})
        {
          CAPTURE(p, q);
          block.evaluateOrbitalGeneratorToAcceptedReference(virtual_positions, 0, particle, p, q,
                                                           actual, orbital_derivatives);
          CHECK(actual.data() == output_storage);
          CHECK(orbital_derivatives.data() == derivative_storage);
          constexpr double epsilon = 1. / 65536; // dyadic column secant, not log differentiation
          for (int term = 0; term < 4; ++term)
          {
            const auto& configuration = block.getConfiguration(term);
            const auto oracle = [&](double theta) {
              return generated_determinant(configuration, moved == 0 ? replacement : 1,
                                           moved == 1 ? replacement : 2, p, q, theta);
            };
            CHECK(actual[term] == base_ratios[term]);
            CHECK(std::real(orbital_derivatives[term]) == Catch::Approx(oracle(0).second / 3).margin(3e-13));
            CHECK(std::real(orbital_derivatives[term]) == Catch::Approx(
                (oracle(epsilon).first - oracle(-epsilon).first) / (6 * epsilon)).margin(3e-11));
            const bool occupied_p = std::find(configuration.occup.begin(), configuration.occup.end(), p) != configuration.occup.end();
            const bool occupied_q = std::find(configuration.occup.begin(), configuration.occup.end(), q) != configuration.occup.end();
            if (occupied_p == occupied_q) CHECK(orbital_derivatives[term] == QMCTraits::ValueType{});
            CHECK(block.getRatiosToRefDet()[term] == original_ratios[term]);
          }
          if (moved == 0 && replacement == 0 && p == 0 && q == 2)
          {
            const auto node = mappings[spin][2]; // occupied {1,2}, but substituted {1,0} is nonzero
            CHECK(actual[node] == QMCTraits::ValueType{});
            CHECK(std::real(orbital_derivatives[node]) == Catch::Approx(4. / 3).margin(3e-13));
          }
          if (moved == 0 && replacement == 0 && p == 1 && q == 2)
          {
            const auto ref = mappings[spin][0], alt = mappings[spin][1];
            // Stationary alt/ref=1/3 and CI1=-6 contract to -2. This is
            // the raw CI numerator with fixed reference product, NOT d(PsiQ/PsiR).
            CHECK(determinant(configurations[0], 0, 2) - 2 * determinant(configurations[1], 0, 2) == 0);
            CHECK(std::real(actual[ref] - 2 * actual[alt]) == Catch::Approx(0).margin(3e-13));
            CHECK(std::real(orbital_derivatives[ref] - 2 * orbital_derivatives[alt]) ==
                  Catch::Approx(10. / 3).margin(3e-13));
          }
          CHECK(block.getLogValueRefDet() == original_log);
        }
        const auto ref = mappings[spin][0], alt = mappings[spin][1];
        const double stationary = std::real(spins[1 - spin]->getRatiosToRefDet()[mappings[1 - spin][1]]);
        const double old_term = std::real(original_ratios[alt]) * stationary;
        const double new_term = std::real(actual[alt]) * stationary;
        const double denominator = 1 - 6 * old_term;
        const double ratio = (std::real(actual[ref]) - 6 * new_term) / denominator;
        const double derivative = (new_term - ratio * old_term) / denominator;
        if ((moved == 0 && replacement == -2) || (moved == 1 && replacement == -1))
        {
          CHECK(actual[ref] == QMCTraits::ValueType{});
          CHECK(std::abs(ratio) > 1); // zero virtual reference, nonzero total CI
        }
        if (replacement == 1 + moved)
        {
          CHECK(ratio == Catch::Approx(1).margin(2e-13));
          CHECK(derivative == Catch::Approx(0).margin(2e-13));
        }
        if (moved == 0 && replacement == 0)
        {
          CHECK(actual[mappings[spin][3]] == QMCTraits::ValueType{});
          // Both spins have accepted alt/ref=1/3. C1=-6 makes Psi(Q)=0
          // while Psi(R)/reference_product=1/3 and dRatio/dC1=2/3.
          CHECK(ratio == Catch::Approx(0).margin(2e-13));
          CHECK(derivative == Catch::Approx(2. / 3).epsilon(2e-13));
          const double epsilon = 1e-6;
          const auto dense_ratio = [&](double coefficient) {
            return (4. / 3 + coefficient * (2. / 9)) / (1 + coefficient / 9);
          };
          CHECK(derivative == Catch::Approx((dense_ratio(-6 + epsilon) - dense_ratio(-6 - epsilon)) /
              (2 * epsilon)).epsilon(2e-8));
        }
      }
      CHECK_THROWS_AS(block.evaluateDetsToAcceptedReference(virtual_positions, 0, 2 * spin - 1, actual),
                      std::invalid_argument);
      for (const auto [p, q] : {std::pair{-1, 2}, std::pair{1, 1}, std::pair{2, 1}, std::pair{0, 4}})
        CHECK_THROWS_AS(block.evaluateOrbitalGeneratorToAcceptedReference(
            virtual_positions, 0, particle, p, q, actual, orbital_derivatives), std::invalid_argument);
      CHECK_THROWS_AS(block.evaluateOrbitalGeneratorToAcceptedReference(
          virtual_positions, 0, particle, 0, 2, actual, actual), std::invalid_argument);
      Vector<QMCTraits::ValueType> attached(actual.data(), actual.size());
      CHECK_THROWS_AS(block.evaluateOrbitalGeneratorToAcceptedReference(
          virtual_positions, 0, particle, 0, 2, attached, orbital_derivatives), std::invalid_argument);
      virtual_positions.R[0][0] = std::numeric_limits<double>::quiet_NaN();
      CHECK_THROWS_AS(block.evaluateDetsToAcceptedReference(virtual_positions, 0, particle, actual), std::domain_error);
      CHECK_THROWS_AS(block.evaluateOrbitalGeneratorToAcceptedReference(
          virtual_positions, 0, particle, 0, 2, actual, orbital_derivatives), std::domain_error);
      virtual_positions.R[0] = {3, 0, 0};
      block.evaluateOrbitalGeneratorToAcceptedReference(virtual_positions, 0, particle, 0, 2,
                                                       actual, orbital_derivatives);
      block.evaluateDetsToAcceptedReference(virtual_positions, 0, particle, actual);
      block.evaluateDetsForPtclMove(virtual_positions, 0, particle);
      CHECK(block.getRefDetRatio() == regular_reference);
      for (int term = 0; term < 4; ++term)
      {
        CHECK(block.getNewRatiosToRefDet()[term] == regular_ratios[term]);
        CHECK(block.getRatiosToRefDet()[term] == original_ratios[term]);
      }
      CHECK(block.getLogValueRefDet() == original_log);
      for (int i = 0; i < 4; ++i) CHECK(accepted.R[i] == original_positions[i]);
    }

  // Component foundation only: explicit private mode must govern actual energy
  // VP ratios and the physical dCI row through the very same scalar helper.
  std::vector<std::unique_ptr<MultiDiracDeterminant>> determinants;
  for (auto& spin : spins) determinants.push_back(std::move(spin));
  MultiSlaterDetTableMethod source(accepted, std::move(determinants), false);
  auto maps = std::make_unique<std::vector<std::vector<std::size_t>>>(2);
  for (int spin = 0; spin < 2; ++spin) (*maps)[spin] = {mappings[spin][0], mappings[spin][1]};
  auto coefficients = std::make_unique<std::vector<QMCTraits::ValueType>>(
      std::initializer_list<QMCTraits::ValueType>{1, -6});
  auto variables = std::make_unique<optimize::VariableSet>();
  variables->insert("CI1", -6, true, optimize::LINEAR_P);
  variables->resetIndex();
  auto active = *variables;
  source.initialize(std::move(maps), std::move(coefficients), std::move(variables), nullptr, true, true);
  source.checkOutVariables(active);
  const auto evaluate = [](MultiSlaterDetTableMethod& component, ParticleSet& particles) {
    ParticleSet::ParticleGradient g(particles.getTotalNum());
    ParticleSet::ParticleLaplacian l(particles.getTotalNum());
    g = QMCTraits::GradType{}; l = QMCTraits::ValueType{};
    return component.evaluateLog(particles, g, l);
  };
  const auto source_log = evaluate(source, accepted);
  // Original x-only physical model: Psi=3, G_x={0,2,0,2}, log-L=-2,
  // hence kinetic energy is zero despite zero derivative references.
  accepted.G = QMCTraits::GradType{};
  accepted.L = QMCTraits::ValueType{};
  source.evaluateLog(accepted, accepted.G, accepted.L);
  for (int electron = 0; electron < 4; ++electron)
  {
    CHECK(std::real(accepted.G[electron][0]) == Catch::Approx(electron % 2 ? 2 : 0).margin(3e-13));
    for (int dimension = 1; dimension < 3; ++dimension)
      CHECK(std::real(accepted.G[electron][dimension]) == Catch::Approx(0).margin(3e-13));
    CHECK(std::real(accepted.L[electron]) == Catch::Approx(-2).margin(3e-13));
  }
  checkClose(kineticEnergy(accepted), 0);
  ParticleSet owned(accepted), sibling_particles(accepted), descendant_particles(accepted);
  auto owned_component = source.makeClone(owned), sibling_component = source.makeClone(sibling_particles);
  auto& private_msd = dynamic_cast<MultiSlaterDetTableMethod&>(*owned_component);
  auto& sibling = dynamic_cast<MultiSlaterDetTableMethod&>(*sibling_component);
  private_msd.isolateOptimizationState();
  private_msd.enableAcceptedReferenceVirtualRatios();
  private_msd.checkOutVariables(active);
  evaluate(private_msd, owned);
  evaluate(sibling, sibling_particles);
  CHECK_FALSE(source.usesAcceptedReferenceVirtualRatios());
  CHECK_FALSE(sibling.usesAcceptedReferenceVirtualRatios());
  CHECK(private_msd.usesAcceptedReferenceVirtualRatios());
  CHECK(source.get_C().data() == sibling.get_C().data());
  CHECK(source.get_C().data() != private_msd.get_C().data());
  auto descendant_component = private_msd.makeClone(descendant_particles);
  CHECK(dynamic_cast<MultiSlaterDetTableMethod&>(*descendant_component).usesAcceptedReferenceVirtualRatios());
  dynamic_cast<MultiSlaterDetTableMethod&>(*descendant_component).isolateOptimizationState();
  VirtualParticleSet ordinary(accepted), vp(owned);
  ordinary.makeMoves(accepted, 0, {{2, 0, 0}});
  std::vector<QMCTraits::ValueType> ordinary_before(1), ordinary_after(1), physical(4);
  source.evaluateRatios(ordinary, ordinary_before);
  const std::vector<ParticleSet::PosType> displacements{{-3, 0, 0}, {-1, 0, 0}, {0, 0, 0}, {2, 0, 0}};
  vp.makeMoves(owned, 0, displacements);
  private_msd.evaluateRatios(vp, physical);
  for (int point = 0; point < 4; ++point)
  {
    std::array<QMCTraits::ValueType, 2> row{0, 123};
    const auto ratio = private_msd.evaluateAcceptedReferenceRatioDerivatives(vp, point, 0, active,
        {row.data(), row.size()});
    CHECK(ratio == physical[point]);
    CHECK(row[1] == QMCTraits::ValueType{123});
    const double x = vp.R[point][0];
    const double a = (4 - x * x) / 3, b = (2 - x) / 9;
    const auto oracle = [=](double c) { return (a + c * b) / (1 + c / 9); };
    CHECK(std::real(ratio) == Catch::Approx(oracle(-6)).epsilon(2e-13).margin(2e-13));
    const double epsilon = 1e-6;
    CHECK(std::real(row[0]) == Catch::Approx((oracle(-6 + epsilon) - oracle(-6 - epsilon)) /
        (2 * epsilon)).epsilon(2e-8).margin(2e-9));
    if (point == 0) CHECK(std::real(ratio) == Catch::Approx(-8).epsilon(2e-13));
    if (point == 1)
    {
      CHECK(std::real(ratio) == Catch::Approx(0).margin(2e-13));
      CHECK(std::real(row[0]) == Catch::Approx(2. / 3).epsilon(2e-13));
    }
    // Independent actual coefficient perturbations refresh accepted denominators.
    std::array<double, 2> finite_difference;
    for (int sign = 0; sign < 2; ++sign)
    {
      active[0] = -6 + (sign == 0 ? epsilon : -epsilon);
      private_msd.resetParametersExclusive(active);
      evaluate(private_msd, owned);
      std::array<QMCTraits::ValueType, 1> scratch{};
      finite_difference[sign] = std::real(private_msd.evaluateAcceptedReferenceRatioDerivatives(
          vp, point, 0, active, {scratch.data(), scratch.size()}));
    }
    CHECK(std::real(row[0]) == Catch::Approx((finite_difference[0] - finite_difference[1]) / (2 * epsilon))
        .epsilon(2e-8).margin(2e-9));
    active[0] = -6;
    private_msd.resetParametersExclusive(active);
    evaluate(private_msd, owned);
  }
  CHECK_THROWS_AS(private_msd.evaluateAcceptedReferenceRatioDerivatives(vp, 0, 0, active, {}), std::invalid_argument);
  std::array<QMCTraits::ValueType, 1> row{};
  vp.makeMoves(owned, 2, displacements);
  private_msd.evaluateRatios(vp, physical);
  CHECK(private_msd.evaluateAcceptedReferenceRatioDerivatives(vp, 1, 2, active, {row.data(), 1}) == physical[1]);
  CHECK(std::real(row[0]) == Catch::Approx(2. / 3).epsilon(2e-13));
  vp.makeMoves(owned, 0, displacements);
  CHECK_THROWS_AS(source.evaluateAcceptedReferenceRatioDerivatives(vp, 0, 0, active, {row.data(), 1}),
                  std::invalid_argument);
  CHECK_THROWS_AS(private_msd.evaluateAcceptedReferenceRatioDerivatives(vp, 0, 4, active, {row.data(), 1}),
                  std::invalid_argument);
  // The original 1/3 ratios can leave a tiny valid nonzero total at C1=-9.
  // Use binary-exact 1/4 ratios only for this accepted-zero negative.
  owned.R[1][0] = owned.R[3][0] = 3;
  owned.update();
  active[0] = -16;
  private_msd.resetParametersExclusive(active);
  evaluate(private_msd, owned);
  CHECK_THROWS_AS(private_msd.evaluateRatios(vp, physical), std::domain_error);
  owned.R[1][0] = owned.R[3][0] = 2;
  owned.update();
  active[0] = std::numeric_limits<double>::quiet_NaN();
  private_msd.resetParametersExclusive(active);
  evaluate(private_msd, owned);
  CHECK_THROWS_AS(private_msd.evaluateRatios(vp, physical), std::domain_error);
  active[0] = -6;
  private_msd.resetParametersExclusive(active);
  evaluate(private_msd, owned);
  vp.makeMoves(owned, 0, displacements);
  private_msd.evaluateRatios(vp, physical);
  CHECK(std::real(physical[1]) == Catch::Approx(0).margin(2e-13));
  CHECK(private_msd.fixedCoefficientReference() == QMCTraits::ValueType{1});
  CHECK(source.get_C()[1] == QMCTraits::ValueType{-6});
  CHECK(sibling.get_C()[1] == QMCTraits::ValueType{-6});
  CHECK(evaluate(source, accepted) == source_log);
  source.evaluateRatios(ordinary, ordinary_after);
  CHECK(ordinary_before == ordinary_after);

  // Two signed, nontrivial CSFs represent the same physical nodal polynomial:
  // 6.5*(Dref-Dalt) - 5.5*(Dref+Dalt) = Dref-12*Dalt.
  // Accepted x={1,3} gives binary-exact alt/ref=1/4 in each spin.
  ParticleSet csf_accepted(accepted);
  csf_accepted.R[1][0] = csf_accepted.R[3][0] = 3;
  csf_accepted.update();
  std::vector<std::unique_ptr<MultiDiracDeterminant>> csf_dets;
  for (int spin = 0; spin < 2; ++spin)
    csf_dets.push_back(std::make_unique<MultiDiracDeterminant>(source.getDet(spin)));
  MultiSlaterDetTableMethod csf_source(accepted, std::move(csf_dets), false);
  auto csf_maps = std::make_unique<std::vector<std::vector<std::size_t>>>(2);
  for (int spin = 0; spin < 2; ++spin)
    (*csf_maps)[spin] = {mappings[spin][0], mappings[spin][1], mappings[spin][0], mappings[spin][1]};
  auto csf_data = std::make_unique<CSFData>();
  csf_data->coeffs = {6.5, -5.5}; csf_data->dets_per_csf = {2, 2};
  csf_data->expansion = {1, -1, 1, 1};
  auto csf_variables = std::make_unique<optimize::VariableSet>();
  csf_variables->insert("CSF1", -5.5, true, optimize::LINEAR_P);
  csf_variables->resetIndex();
  auto csf_active = *csf_variables;
  csf_source.initialize(std::move(csf_maps),
      std::make_unique<std::vector<QMCTraits::ValueType>>(
          std::initializer_list<QMCTraits::ValueType>{6.5, -6.5, -5.5, -5.5}),
      std::move(csf_variables), std::move(csf_data), true, true);
  csf_source.checkOutVariables(csf_active);
  const auto csf_source_log = evaluate(csf_source, csf_accepted);
  ParticleSet csf_private_p(csf_accepted);
  auto csf_private_owner = csf_source.makeClone(csf_private_p);
  auto& csf_private = dynamic_cast<MultiSlaterDetTableMethod&>(*csf_private_owner);
  csf_private.isolateOptimizationState(); csf_private.enableAcceptedReferenceVirtualRatios();
  evaluate(csf_private, csf_private_p);
  VirtualParticleSet csf_vp(csf_private_p);
  for (int spin = 0; spin < 2; ++spin)
  {
    csf_vp.makeMoves(csf_private_p, 2 * spin, {{-4, 0, 0}, {-1, 0, 0}, {0, 0, 0}, {2, 0, 0}});
    std::vector<QMCTraits::ValueType> ratios(4);
    csf_private.evaluateRatios(csf_vp, ratios);
    for (int point = 0; point < 4; ++point)
    {
      std::array<QMCTraits::ValueType, 1> derivative{};
      const auto actual = csf_private.evaluateAcceptedReferenceRatioDerivatives(csf_vp, point, 2 * spin,
          csf_active, {derivative.data(), 1});
      const double x = csf_vp.R[point][0];
      const auto dense_ratio = [=](double c) {
        // Independently occupied 2x2 polynomials, including both spin blocks.
        const double ref = 8 * (9 - x * x), alt = 2 * (3 - x);
        return ((6.5 + c) * ref + (-6.5 + c) * alt) / (64 * (6.5 + c) + 4 * (-6.5 + c));
      };
      CHECK(std::real(actual) == Catch::Approx(dense_ratio(-5.5)).margin(2e-13));
      CHECK(actual == ratios[point]);
      CHECK(std::real(derivative[0]) == Catch::Approx(
          (dense_ratio(-5.5 + 1e-6) - dense_ratio(-5.5 - 1e-6)) / 2e-6).epsilon(2e-8).margin(2e-9));
      if (point == 0) CHECK(std::real(actual) == Catch::Approx(-9).margin(2e-13));
      if (point == 1)
      {
        CHECK(actual == QMCTraits::ValueType{});
        CHECK(std::real(derivative[0]) == Catch::Approx(39. / 8).epsilon(2e-13));
      }
    }
  }
  CHECK_FALSE(csf_source.usesAcceptedReferenceVirtualRatios());
  CHECK(evaluate(csf_source, csf_accepted) == csf_source_log);
  CHECK(csf_source.fixedCoefficientReference() == QMCTraits::ValueType{6.5});

  const bool csf = GENERATE(false, true);
  INFO("CSF expansion=" << csf);
  // Actual Be BFD capture on the polynomial DETS/CSF model: octahedral knots
  // include virtual-reference nodes and CI cancellation, with finite dCI.
  ParticleSet p(cell), ions(cell);
  ions.setName("ion0"); ions.create({1}); ions.R[0] = {0, 0, 0};
  auto& isp = ions.getSpeciesSet();
  const int be = isp.addSpecies("Be"), z = isp.addAttribute("charge");
  isp(z, be) = 2; ions.resetGroups(); ions.update();
  p.setName("e"); p.create({2, 2}); p.R = accepted.R;
  if (csf) p.R[1][0] = p.R[3][0] = 3;
  p.R[2][2] = p.R[3][2] = 0.5; // avoid opposite-spin electron coincidence
  auto& esp = p.getSpeciesSet();
  const int u = esp.addSpecies("u"), d = esp.addSpecies("d"), mass = esp.addAttribute("mass");
  esp(mass, u) = esp(mass, d) = 1;
  p.resetGroups(); p.addTable(ions); p.update();
  auto wf = makeBsplineJ2WaveFunction(p, {0.12, -0.08, 0.04, -0.01});
  wf->addComponent((csf ? csf_source : source).makeClone(p));
  auto& live_msd = wf->findMSD().front().get();
  live_msd.isolateOptimizationState();
  live_msd.enableAcceptedReferenceVirtualRatios();
  for (int spin = 0; spin < 2; ++spin)
    dynamic_cast<PolynomialSPO&>(*live_msd.getDet(spin).getPhi()).transverse_factor = true;
  p.update(); // J2 construction added its AA distance table after the ion table.
  wf->evaluateLog(p);
  // The legacy VGL implementation inverse-updates derivative reference rows;
  // this ECP fixture must not use a zero reference gradient or Laplacian.
  for (int spin = 0; spin < 2; ++spin)
    for (int electron = 0; electron < 2; ++electron)
    {
      const auto& block = live_msd.getDet(spin);
      CHECK(std::real(block.getGrads()(0, electron)[0]) == Catch::Approx(
          csf ? (electron == 0 ? -0.25 : 0.75) : (electron == 0 ? -2. / 3 : 4. / 3)));
      for (int dim = 1; dim < 3; ++dim)
        CHECK(std::real(block.getGrads()(0, electron)[dim]) == Catch::Approx(spin == 0 ? 1 : 2. / 3));
      CHECK(std::real(block.getLapls()(0, electron)) == Catch::Approx(
          csf ? (electron == 0 ? -0.25 : 0.25) : (electron == 0 ? -2. / 3 : 2. / 3)));
    }
  const auto live_log = wf->getLogPsi();
  const auto live_energy = kineticEnergy(p);
  const auto positions = p.R;
  LegacyWaveFunctionAdapter adapter("conventional", *wf, adapterOptions());
  ParticleSet unsupported_j3_particles(p);
  auto unsupported_j3 = wf->makeClone(unsupported_j3_particles);
  addPolynomialJ3(*unsupported_j3, unsupported_j3_particles, ions);
  CHECK_THROWS_WITH(adapter.validateNonLocalECPProfile(*unsupported_j3, unsupported_j3_particles),
      Catch::Matchers::ContainsSubstring("fixed single Slater without active orbitals"));
  const auto snapshot = std::make_shared<const StructuredParameterSnapshot>(adapter.snapshotParameters());
  REQUIRE(snapshot->values.size() == 5);
  ParticleSet replay_p(p);
  auto replay_wf = wf->makeClone(replay_p);
  replay_wf->findMSD().front().get().isolateOptimizationState();
  replay_p.update();
  class FixedDETSReplay final : public LegacyWaveFunctionReplay
  {
  public:
    FixedDETSReplay(TrialWaveFunction& wf, ParticleSet& p) : wf(wf), p(p), positions(p.R) {}
    std::size_t sampleCount() const noexcept override { return 1; }
    LegacyWaveFunctionReplaySample prepareSample(std::size_t index) override
    {
      REQUIRE(index == 0);
      p.R = positions; p.update();
      return {wf, p};
    }
    TrialWaveFunction& wf;
    ParticleSet& p;
    ParticleSet::ParticlePos positions;
  };
  auto replay = std::make_shared<FixedDETSReplay>(*replay_wf, replay_p);
  auto op = adapter.makeDerivativeOperator(replay, snapshot, 0, 0);
  auto consumer = adapter.makeNonLocalECPDerivativeConsumer(replay, snapshot, 0, 0);
  ECPComponentBuilder builder("Be", OHMMS::Controller, 3, 1);
  const auto asset = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() /
      "tests/pseudopotentials_for_tests/Be.BFD.xml";
  REQUIRE(builder.read_pp_file(asset.string()));
  auto nonlocal = std::make_unique<NonLocalECPotential>(ions, p, false, true);
  auto* nl = nonlocal.get();
  nonlocal->addComponent(0, std::move(builder.pp_nonloc));
  auto local = std::make_unique<LocalECPotential>(ions, p);
  auto* loc = local.get();
  local->add(0, std::move(builder.pp_loc), builder.Zeff);
  QMCHamiltonian ham;
  ham.addOperator(std::make_unique<BareKineticEnergy>(p), "Kinetic");
  ham.addOperator(std::move(local), "LocalECP");
  ham.addOperator(std::move(nonlocal), "NonLocalECP");
  ham.addObservables(p);
  class CountZero final : public FakeRandom<QMCTraits::FullPrecRealType>
  {
  public:
    QMCTraits::FullPrecRealType operator()() override { ++draws; return 0; }
    std::size_t draws = 0;
  } rng;
  nl->setRandomGenerator(&rng);
  ResourceCollection pr("md_ecp_P"), wr("md_ecp_W"), hr("md_ecp_H");
  p.createResource(pr); wf->createResource(wr); ham.createResource(hr);
  RefVectorWithLeader<ParticleSet> ps(p, {p});
  RefVectorWithLeader<TrialWaveFunction> wfs(*wf, {*wf});
  RefVectorWithLeader<QMCHamiltonian> hs(ham, {ham});
  const auto ecp_mask = localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);
  const auto total_mask = ecp_mask | localEnergyTermBit(LocalEnergyTerm::KINETIC);
  const std::vector<DerivativeValue> unit{1};
  const VJPCoefficientChannel capture{"capture", DerivativeProduct::LOCAL_ENERGY_VJP,
      coefficientsFor(*op, unit), ecp_mask};
  RecordingParameterSink idle;
  QMCHamiltonian::StreamingNonLocalECPResult captured;
  {
    ResourceCollectionTeamLock<ParticleSet> pl(pr, ps);
    ResourceCollectionTeamLock<TrialWaveFunction> wl(wr, wfs);
    ResourceCollectionTeamLock<QMCHamiltonian> hl(hr, hs);
    captured = QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(hs, wfs, ps, *consumer, {&capture, 1}, idle);
  }
  const auto draws = rng.draws;
  CHECK(draws > 0);
  auto record = consumer->releaseJVPReplay();
  const std::vector<NonLocalECPReplayPoint> points(record->points().begin(), record->points().end());
  REQUIRE(!points.empty());
  CHECK(std::any_of(points.begin(), points.end(), [](const auto& q) { return q.complete_ratio == DerivativeValue{}; }));
  bool nonzero_ci_at_zero_ratio = false;
  for (const auto& q : points)
  {
    if (q.complete_ratio != DerivativeValue{}) continue;
    replay->prepareSample(0);
    adapter.installSnapshot(*replay_wf, replay_p, *snapshot);
    for (int dim = 0; dim < 3; ++dim) replay_p.R[q.electron][dim] = q.replacement_position[dim];
    replay_p.update();
    std::array<QMCTraits::ValueType, 5> derivative{};
    replay_wf->findMSD().front().get().evaluateAcceptedReferenceRatioDerivatives(
        replay_p, q.electron, q.electron, adapter.registrationVariables(), {derivative.data(), derivative.size()});
    nonzero_ci_at_zero_ratio |= std::any_of(derivative.begin(), derivative.end(),
        [](auto value) { return std::abs(value) > 0.1; });
  }
  CHECK(nonzero_ci_at_zero_ratio);
  op->bindNonLocalECPJVPReplay(std::move(record));
  const double local_value = loc->evaluate(p);
  // Independent occupied 2x2 determinant polynomials, not the bordered helper.
  auto oracle_points = points;
  auto oracle_positions = positions;
  double oracle_local_value = local_value;
  const auto fixed_energy = [&](const StructuredParameterSnapshot& candidate) {
    ParticleSet trial(p);
    auto twf = wf->makeClone(trial);
    twf->findMSD().front().get().isolateOptimizationState();
    trial.R = oracle_positions;
    trial.update(); adapter.installSnapshot(*twf, trial, candidate);
    const auto& expanded = twf->findMSD().front().get().get_C();
    const double ref_coefficient = std::real(expanded[0] + (csf ? expanded[2] : QMCTraits::ValueType{}));
    const double alt_coefficient = std::real(expanded[1] + (csf ? expanded[3] : QMCTraits::ValueType{}));
    const auto jlog = [&](TrialWaveFunction& state, ParticleSet& particles) {
      double value = 0;
      particles.G = QMCTraits::GradType{}; particles.L = QMCTraits::ValueType{};
      for (auto& component : state.getOrbitals())
        if (!component->isFermionic()) value += component->evaluateLog(particles, particles.G, particles.L).real();
      return value;
    };
    const double accepted_j = jlog(*twf, trial);
    trial.R = oracle_positions; trial.update(); adapter.installSnapshot(*twf, trial, candidate);
    DerivativeValue energy = kineticEnergy(trial) + oracle_local_value;
    const auto determinant = [](const ParticleSet::ParticlePos& r, int spin, bool alt) {
      const double x = r[2 * spin][0], y = r[2 * spin + 1][0];
      return alt ? y - x : y * y - x * x;
    };
    const double accepted_reference = determinant(oracle_positions, 0, false) * determinant(oracle_positions, 1, false);
    const double accepted_alternative = determinant(oracle_positions, 0, true) * determinant(oracle_positions, 1, true);
    for (const auto& q : oracle_points)
    {
      auto r = oracle_positions;
      for (int dim = 0; dim < 3; ++dim) r[q.electron][dim] = q.replacement_position[dim];
      const double common_ratio = (1 + r[q.electron][1] + r[q.electron][2]) /
          (1 + oracle_positions[q.electron][1] + oracle_positions[q.electron][2]);
      const double ratio = common_ratio * (ref_coefficient * determinant(r, 0, false) * determinant(r, 1, false) +
          alt_coefficient * determinant(r, 0, true) * determinant(r, 1, true)) /
          (accepted_reference * ref_coefficient + accepted_alternative * alt_coefficient);
      trial.R = r; trial.update();
      energy += q.bare_weight * ratio * std::exp(jlog(*twf, trial) - accepted_j);
    }
    return energy;
  };
  CHECK(fixed_energy(*snapshot).real() == Catch::Approx(captured.local_energies[0]).epsilon(3e-12).margin(3e-12));
  const VJPCoefficientChannel total{"total", DerivativeProduct::LOCAL_ENERGY_VJP,
      coefficientsFor(*op, unit), total_mask};
  RecordingParameterSink transpose, hermitian;
  op->applyVJPs({&total, 1}, DerivativeAdjoint::TRANSPOSE, transpose);
  op->applyVJPs({&total, 1}, DerivativeAdjoint::HERMITIAN, hermitian);
  std::vector<DerivativeValue> direction(5);
  DerivativeValue dual{};
  for (std::size_t parameter = 0; parameter < 5; ++parameter)
  {
    auto plus = *snapshot, minus = *snapshot;
    plus.values[parameter] += 1e-5; minus.values[parameter] -= 1e-5;
    const auto fd = (fixed_energy(plus) - fixed_energy(minus)) / 2e-5;
    CHECK(transpose.result()[0][parameter].real() == Catch::Approx(fd.real()).epsilon(3e-6).margin(3e-8));
    checkClose(hermitian.result()[0][parameter], std::conj(transpose.result()[0][parameter]));
    direction[parameter] = 0.1 * (parameter + 1);
    dual += direction[parameter] * transpose.result()[0][parameter];
  }
  RecordingSampleSink jvp;
  op->applyLocalEnergyJVP({adapter.parameterSchema(), snapshot->version, {direction.data(), direction.size()}}, total_mask, jvp);
  checkClose(jvp.result()[0], dual);
  CHECK(op->storageDiagnostics().real_parameter_vectors == 2);
  CHECK(rng.draws == draws);
  for (int electron = 0; electron < p.getTotalNum(); ++electron)
    CHECK(p.R[electron] == positions[electron]);
  wf->evaluateLog(p);
  CHECK(wf->getLogPsi() == live_log);
  checkClose(kineticEnergy(p), live_energy);
  CHECK_FALSE(source.usesAcceptedReferenceVirtualRatios());
  CHECK_FALSE(csf_source.usesAcceptedReferenceVirtualRatios());
  CHECK(evaluate(csf_source, csf_accepted) == csf_source_log);

  // The physical octahedral grid cannot hit q_x=+-2 (DETS) or +-3 (CSF) for the radius-1 first
  // electron. Exercise that exact reference node in a separately labeled
  // synthetic replay, not by changing or relabeling the accepted PP grid.
  NonLocalECPReplayPoint node{};
  const double reference_node = csf ? -3 : -2;
  node.sample = 0; node.electron = 0; node.replacement_position = {reference_node, 0, 0}; node.bare_weight = 1;
  oracle_points = {node};
  node.complete_ratio = fixed_energy(*snapshot) - live_energy - local_value;
  REQUIRE(node.complete_ratio != DerivativeValue{});
  CHECK(node.replacement_position[0] * node.replacement_position[0] == positions[1][0] * positions[1][0]);
  auto node_op = adapter.makeDerivativeOperator(replay, snapshot, 1, 0);
  auto node_capture = adapter.makeNonLocalECPDerivativeConsumer(replay, snapshot, 1, 0);
  const VJPCoefficientChannel node_channel{"node", DerivativeProduct::LOCAL_ENERGY_VJP,
      coefficientsFor(*node_op, unit), ecp_mask};
  const NonLocalECPDerivativeContext context{adapter.parameterSchema().providerId(),
      adapter.parameterSchema().fingerprint(), snapshot->version, 1, 0, 1, 1, 456};
  node_capture->begin(context, {&node_channel, 1}, idle);
  const std::vector<std::size_t> offsets{0, 1};
  const std::vector<VirtualParticleBatch::Segment> segments{{0, 0, true, 0}};
  const std::vector<ParticleSet::PosType> replacements{{reference_node, 0, 0}};
  VirtualParticleBatch tile(1, offsets, segments, replacements);
  const QMCTraits::ValueType bare{1}, ratio{node.complete_ratio.real()};
  node_capture->consume({tile, {&bare, 1}, {&ratio, 1}, {}, 0, 456});
  node_capture->end();
  node_op->bindNonLocalECPJVPReplay(node_capture->releaseJVPReplay());
  const VJPCoefficientChannel node_total{"node_total", DerivativeProduct::LOCAL_ENERGY_VJP,
      coefficientsFor(*node_op, unit), total_mask};
  RecordingParameterSink node_rows;
  node_op->applyVJPs({&node_total, 1}, DerivativeAdjoint::TRANSPOSE, node_rows);
  for (std::size_t parameter = 0; parameter < 5; ++parameter)
  {
    auto plus = *snapshot, minus = *snapshot;
    plus.values[parameter] += 1e-5; minus.values[parameter] -= 1e-5;
    const auto fd = (fixed_energy(plus) - fixed_energy(minus)) / 2e-5;
    CHECK(node_rows.result()[0][parameter].real() == Catch::Approx(fd.real()).epsilon(3e-6).margin(3e-8));
  }
  CHECK(rng.draws == draws);
  if (!csf)
  {
    // Four independently captured configurations make the centered factor
    // nontrivial. All retained operators borrow one evaluator/shared P scratch.
    constexpr std::size_t n = 5, m = 4;
    const std::array<double, m> shifts{-0.15, -0.05, 0.08, 0.2}, weights{1, 2, 1, 2};
    std::array<DerivativeValue, m> energies{};
    std::array<std::array<double, n>, m> scores{}, del{}; // tiny dense oracle only
    std::array<double, n> mean{};
    double mean_energy = 0;
    std::vector<std::unique_ptr<StreamingDerivativeOperator>> operators;
    std::vector<ScoreCovarianceBatch> batches;
    for (std::size_t sample = 0; sample < m; ++sample)
    {
      p.R = positions;
      p.R[0][0] += shifts[sample];
      p.R[1][2] += 0.03 * sample;
      p.update();
      wf->evaluateLog(p);
      auto sample_replay = std::make_shared<FixedDETSReplay>(*replay_wf, replay_p);
      sample_replay->positions = p.R;
      auto products = adapter.makeDerivativeOperator(sample_replay, snapshot, sample + 2, sample);
      auto capture_consumer = adapter.makeNonLocalECPDerivativeConsumer(sample_replay, snapshot, sample + 2, sample);
      const VJPCoefficientChannel sample_channel{"sample_capture", DerivativeProduct::LOCAL_ENERGY_VJP,
          coefficientsFor(*products, unit), ecp_mask};
      {
        ResourceCollectionTeamLock<ParticleSet> pl(pr, ps);
        ResourceCollectionTeamLock<TrialWaveFunction> wl(wr, wfs);
        ResourceCollectionTeamLock<QMCHamiltonian> hl(hr, hs);
        energies[sample] = QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(
            hs, wfs, ps, *capture_consumer, {&sample_channel, 1}, idle).local_energies[0];
      }
      auto grid = capture_consumer->releaseJVPReplay();
      oracle_points.assign(grid->points().begin(), grid->points().end());
      oracle_positions = p.R;
      oracle_local_value = loc->evaluate(p);
      products->bindNonLocalECPJVPReplay(std::move(grid));
      CHECK(products->storageDiagnostics().real_parameter_vectors == 2);
      CHECK(products->storageDiagnostics().shared_workspace_identity == op->storageDiagnostics().shared_workspace_identity);
      const std::array<VJPCoefficientChannel, 2> channels{{
          {"score", DerivativeProduct::SCORE_VJP, coefficientsFor(*products, unit), 0},
          {"energy", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(*products, unit), total_mask}}};
      RecordingParameterSink rows;
      products->applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::TRANSPOSE, rows);
      // Physical FD uses the independent polynomial ratio oracle on THIS grid.
      for (std::size_t parameter = 0; parameter < n; ++parameter)
      {
        scores[sample][parameter] = rows.result()[0][parameter].real();
        del[sample][parameter] = rows.result()[1][parameter].real();
        mean[parameter] += weights[sample] * scores[sample][parameter] / 6;
        auto plus = *snapshot, minus = *snapshot;
        plus.values[parameter] += 1e-5;
        minus.values[parameter] -= 1e-5;
        CHECK(del[sample][parameter] == Catch::Approx(
            ((fixed_energy(plus) - fixed_energy(minus)) / 2e-5).real()).epsilon(3e-6).margin(3e-8));
      }
      CHECK(fixed_energy(*snapshot).real() == Catch::Approx(energies[sample].real()).epsilon(3e-12).margin(3e-12));
      mean_energy += weights[sample] * energies[sample].real() / 6;
      batches.push_back({products.get(), {&weights[sample], 1}, {&energies[sample], 1}});
      operators.push_back(std::move(products));
    }
    const auto captured_draws = rng.draws;
    const auto& schema = adapter.parameterSchema();
    ScoreCovarianceFactor factor(schema, snapshot->version, {batches.data(), batches.size()});
    ProjectedInverseIterationFactor pii_factor(factor, total_mask);
    PIIUpdateControl control;
    control.shift_mode = PIIShiftMode::EXPLICIT;
    control.explicit_shift = mean_energy - 1;
    control.damping = 0.3;
    control.learning_rate = 0.01;
    control.maximum_damping_retries = 0;
    control.gmres.restart_length = 16;
    control.gmres.krylov.maximum_iterations = 64;
    control.gmres.krylov.maximum_operator_applications = 256;
    control.gmres.krylov.maximum_seconds = 5;
    std::array<std::array<double, n>, n> matrix{};
    std::array<double, n> rhs{};
    double metric_trace = 0, force_norm = 0;
    for (std::size_t row = 0; row < n; ++row)
    {
      matrix[row][row] = control.damping;
      for (std::size_t sample = 0; sample < m; ++sample)
      {
        const double centered = scores[sample][row] - mean[row];
        const double left = weights[sample] * centered / 6;
        metric_trace += left * centered;
        rhs[row] += left * (energies[sample].real() - mean_energy);
        for (std::size_t column = 0; column < n; ++column)
          matrix[row][column] += left * (del[sample][column] +
              (energies[sample].real() - control.explicit_shift) * (scores[sample][column] - mean[column]));
      }
      force_norm = std::hypot(force_norm, rhs[row]);
    }
    REQUIRE(metric_trace > 1e-8);
    REQUIRE(force_norm > 1e-8);
    ProjectedInverseIterationOperator action(pii_factor, control.explicit_shift, control.damping);
    for (std::size_t column = 0; column < n; ++column)
    {
      std::array<DerivativeValue, n> basis{};
      std::array<DerivativeValue, n> result{};
      basis[column] = 1;
      action.apply({schema, snapshot->version, {basis.data(), n}}, {result.data(), n});
      for (std::size_t row = 0; row < n; ++row)
        CHECK(result[row].real() == Catch::Approx(matrix[row][column]).epsilon(2e-10).margin(2e-10));
    }
    // Independent pivoted dense solve, not another call to the iterative core.
    for (std::size_t column = 0; column < n; ++column)
    {
      std::size_t pivot = column;
      for (std::size_t row = column + 1; row < n; ++row)
        if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) pivot = row;
      REQUIRE(std::abs(matrix[pivot][column]) > 1e-13);
      std::swap(matrix[pivot], matrix[column]);
      std::swap(rhs[pivot], rhs[column]);
      for (std::size_t row = column + 1; row < n; ++row)
      {
        const double multiplier = matrix[row][column] / matrix[column][column];
        for (std::size_t entry = column; entry < n; ++entry)
          matrix[row][entry] -= multiplier * matrix[column][entry];
        rhs[row] -= multiplier * rhs[column];
      }
    }
    std::array<double, n> direction{};
    for (std::size_t reverse = n; reverse > 0; --reverse)
    {
      const auto row = reverse - 1;
      double value = rhs[row];
      for (std::size_t column = row + 1; column < n; ++column) value -= matrix[row][column] * direction[column];
      direction[row] = value / matrix[row][row];
    }
    const auto gradient = factor.scoreCovarianceGradient();
    const ParameterGradientView objective{schema.fingerprint(), snapshot->version, ReductionDomain::GLOBAL,
        {gradient.data(), gradient.size()}};
    ProjectedInverseIterationUpdateRule pii(pii_factor, control);
    MinimumProjectedInverseIterationUpdateRule minpii(pii_factor, control);
    for (const auto& candidate : {pii.propose(schema, *snapshot, objective), minpii.propose(schema, *snapshot, objective)})
    {
      for (std::size_t parameter = 0; parameter < n; ++parameter)
        CHECK(candidate.values[parameter] == Catch::Approx(snapshot->values[parameter] -
            control.learning_rate * direction[parameter]).epsilon(2e-9).margin(2e-11));
      CHECK(isFiniteTrainingReal(fixed_energy(candidate).real()));
    }
    auto plus = *snapshot, minus = *snapshot;
    double expected_derivative = 0;
    for (std::size_t parameter = 0; parameter < n; ++parameter)
    {
      plus.values[parameter] += 1e-5 * direction[parameter];
      minus.values[parameter] -= 1e-5 * direction[parameter];
      expected_derivative += del[m - 1][parameter] * direction[parameter];
    }
    CHECK(((fixed_energy(plus) - fixed_energy(minus)) / 2e-5).real() ==
        Catch::Approx(expected_derivative).epsilon(3e-6).margin(3e-8));
    pii.proposalRejected();
    minpii.proposalRejected();
    CHECK(rng.draws == captured_draws);
    CHECK(adapter.snapshotParameters().values == snapshot->values);
    CHECK(live_msd.fixedCoefficientReference() == QMCTraits::ValueType{1});
    p.R = positions;
    p.update();
    wf->evaluateLog(p);
    CHECK(wf->getLogPsi() == live_log);
    checkClose(kineticEnergy(p), live_energy);
  }
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
        <csf id="CI0" coeff="0.6" occ="211" exctLvl="1">
          <det coeff="0.7071067811865475244" alpha="110" beta="101"/>
          <det coeff="0.7071067811865475244" alpha="101" beta="110"/>
        </csf>
        <csf id="CI1" coeff="0" occ="121" exctLvl="1">
          <det coeff="0.7071067811865475244" alpha="110" beta="011"/>
          <det coeff="0.7071067811865475244" alpha="011" beta="110"/>
        </csf>
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
    REQUIRE(source_msd.get_C().size() == (csf ? 4 : 2));
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
    const auto original_log = source->getLogPsi();
    const auto original_energy = kineticEnergy(electrons);
    const auto* original_spo = &source->getSPOSet("spo");
    const std::size_t active_term = csf ? 2 : 1;
    const double expansion_factor = csf ? std::sqrt(0.5) : 1.0;
    for (double coefficient : {0.0, -0.17})
    {
      CAPTURE(coefficient);
      auto candidate = snapshot;
      candidate.values[0] = coefficient;
      adapter.installSnapshot(*evaluator, private_particles, candidate);
      CHECK(private_msd.get_C()[0] == original_coefficients[0]);
      CHECK(private_msd.fixedCoefficientReference() == 0.6);
      CHECK(private_msd.get_C()[active_term] == makeValue(coefficient * expansion_factor));
      if (csf)
      {
        CHECK(private_msd.get_C()[1] == original_coefficients[1]);
        CHECK(private_msd.get_C()[3] == private_msd.get_C()[2]);
      }
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
      {
        // Mixed CI/J2 inverse iteration uses the same local additive tangent
        // policy as nonlinear models, not LM's centered Ritz-vector retraction.
        constexpr std::size_t n = 9, m = 4;
        auto replay = std::make_shared<ShiftedLegacyReplay>(*evaluator, private_particles,
            std::vector<double>{-0.12, -0.03, 0.07, 0.18});
        auto products = adapter.makeDerivativeOperator(replay,
            std::make_shared<const StructuredParameterSnapshot>(candidate), 0, 0);
        std::array<std::array<double, n>, m> scores{}, del{};
        std::array<DerivativeValue, m> energies{};
        std::array<double, m> weights{1, 2, 1, 2};
        std::array<double, n> mean{};
        double mean_energy = 0;
        auto active = adapter.registrationVariables();
        active[0] = coefficient;
        for (std::size_t sample = 0; sample < m; ++sample)
        {
          replay->prepareSample(sample);
          adapter.installSnapshot(*evaluator, private_particles, candidate);
          energies[sample] = kineticEnergy(private_particles);
          mean_energy += weights[sample] * energies[sample].real() / 6;
          Vector<QMCTraits::ValueType> raw(n), energy_row(n);
          raw = energy_row = 0;
          evaluator->evaluateDerivatives(private_particles, active, raw, energy_row);
          for (std::size_t parameter = 0; parameter < n; ++parameter)
          {
            scores[sample][parameter] = std::real(raw[parameter]);
            del[sample][parameter] = std::real(energy_row[parameter]);
            mean[parameter] += weights[sample] * scores[sample][parameter] / 6;
          }
        }
        const auto& schema = adapter.parameterSchema();
        const ScoreCovarianceBatch batch{products.get(), {weights.data(), m}, {energies.data(), m}};
        ScoreCovarianceFactor factor(schema, candidate.version, {&batch, 1});
        ProjectedInverseIterationFactor pii_factor(factor, localEnergyTermBit(LocalEnergyTerm::KINETIC));
        PIIUpdateControl control;
        control.shift_mode = PIIShiftMode::EXPLICIT;
        control.explicit_shift = mean_energy - 1;
        control.damping = 0.3;
        control.learning_rate = 0.01;
        control.maximum_damping_retries = 0;
        control.gmres.restart_length = 16;
        control.gmres.krylov.maximum_iterations = 64;
        control.gmres.krylov.maximum_operator_applications = 256;
        control.gmres.krylov.maximum_seconds = 5;
        std::vector<std::vector<double>> matrix(n, std::vector<double>(n));
        std::vector<double> rhs(n);
        for (std::size_t row = 0; row < n; ++row)
        {
          matrix[row][row] = control.damping;
          for (std::size_t sample = 0; sample < m; ++sample)
          {
            const double left = weights[sample] * (scores[sample][row] - mean[row]) / 6;
            rhs[row] += left * (energies[sample].real() - mean_energy);
            for (std::size_t column = 0; column < n; ++column)
              matrix[row][column] += left * (del[sample][column] +
                  (energies[sample].real() - control.explicit_shift) * (scores[sample][column] - mean[column]));
          }
        }
        // Same pivoted dense elimination recipe as the independent PII tests.
        for (std::size_t column = 0; column < n; ++column)
        {
          std::size_t pivot = column;
          for (std::size_t row = column + 1; row < n; ++row)
            if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) pivot = row;
          REQUIRE(std::abs(matrix[pivot][column]) > 1e-13);
          std::swap(matrix[pivot], matrix[column]);
          std::swap(rhs[pivot], rhs[column]);
          for (std::size_t row = column + 1; row < n; ++row)
          {
            const double multiplier = matrix[row][column] / matrix[column][column];
            for (std::size_t entry = column; entry < n; ++entry)
              matrix[row][entry] -= multiplier * matrix[column][entry];
            rhs[row] -= multiplier * rhs[column];
          }
        }
        std::vector<double> direction(n);
        for (std::size_t reverse = n; reverse > 0; --reverse)
        {
          const auto row = reverse - 1;
          double value = rhs[row];
          for (std::size_t column = row + 1; column < n; ++column) value -= matrix[row][column] * direction[column];
          direction[row] = value / matrix[row][row];
        }
        const auto gradient = factor.scoreCovarianceGradient();
        const ParameterGradientView objective{schema.fingerprint(), candidate.version, ReductionDomain::GLOBAL,
            {gradient.data(), gradient.size()}};
        ProjectedInverseIterationUpdateRule pii(pii_factor, control);
        MinimumProjectedInverseIterationUpdateRule minpii(pii_factor, control);
        for (const auto& mapped : {pii.propose(schema, candidate, objective), minpii.propose(schema, candidate, objective)})
          for (std::size_t parameter = 0; parameter < n; ++parameter)
            CHECK(mapped.values[parameter] == Catch::Approx(candidate.values[parameter] -
                control.learning_rate * direction[parameter]).epsilon(2e-9).margin(2e-11));
        pii.proposalRejected();
        minpii.proposalRejected();
        for (std::size_t sample = 0; sample < m; ++sample)
        {
          double expected_log = 0, expected_energy = 0;
          for (std::size_t parameter = 0; parameter < n; ++parameter)
          {
            expected_log += scores[sample][parameter] * direction[parameter];
            expected_energy += del[sample][parameter] * direction[parameter];
          }
          double previous_log_error = 0, previous_energy_error = 0;
          for (double step : {1e-3, 5e-4})
          {
            auto plus = candidate, minus = candidate;
            for (std::size_t parameter = 0; parameter < n; ++parameter)
            {
              plus.values[parameter] += step * direction[parameter];
              minus.values[parameter] -= step * direction[parameter];
            }
            replay->prepareSample(sample);
            adapter.installSnapshot(*evaluator, private_particles, plus);
            const double plus_log = evaluator->getLogPsi(), plus_energy = kineticEnergy(private_particles).real();
            adapter.installSnapshot(*evaluator, private_particles, minus);
            const double log_error = std::abs((plus_log - evaluator->getLogPsi()) / (2 * step) - expected_log);
            const double energy_error = std::abs((plus_energy - kineticEnergy(private_particles).real()) / (2 * step) - expected_energy);
            CHECK(std::isfinite(log_error));
            CHECK(std::isfinite(energy_error));
            if (step == 5e-4)
            {
              CHECK(log_error <= 0.4 * previous_log_error + 5e-9);
              CHECK(energy_error <= 0.4 * previous_energy_error + 5e-9);
            }
            previous_log_error = log_error;
            previous_energy_error = energy_error;
            CHECK(private_msd.fixedCoefficientReference() == 0.6);
            CHECK(private_msd.get_C()[0] == original_coefficients[0]);
            if (csf)
            {
              CHECK(private_msd.get_C()[1] == original_coefficients[1]);
              CHECK(private_msd.get_C()[2] == makeValue(minus.values[0] * expansion_factor));
              CHECK(private_msd.get_C()[3] == private_msd.get_C()[2]);
            }
          }
        }
        source->evaluateLog(electrons);
        CHECK(source->getLogPsi() == original_log);
        CHECK(kineticEnergy(electrons) == original_energy);
        CHECK(&source->getSPOSet("spo") == original_spo);
        CHECK(source_msd.get_C() == original_coefficients);
        CHECK(adapter.snapshotParameters().values == snapshot.values);
      }
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
    CHECK(source_msd.fixedCoefficientReference() == 0.6);
    CHECK(source_msd.get_C()[active_term] == makeValue(-0.23 * expansion_factor));
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
      CHECK(source_msd.fixedCoefficientReference() == 0.6);
      CHECK(source_msd.get_C()[active_term] == makeValue(next.values[0] * expansion_factor));
      if (csf)
      {
        CHECK(source_msd.get_C()[1] == original_coefficients[1]);
        CHECK(source_msd.get_C()[3] == source_msd.get_C()[2]);
      }
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

#ifndef QMC_COMPLEX
TEST_CASE("CI-only centered family actions and fixed-C0 ratios", "[legacy_adapter][ci_family_oracle]")
{
  // Preserve the existing AE DETS/CSF cases, then exercise pure DETS ECP.
  const int model = GENERATE(0, 1, 2);
  const bool csf = model == 1, ecp = model == 2;
  CAPTURE(model);
  ParticleSetPool pool(OHMMS::Controller);
  Libxml2Document definition;
  const auto fixture = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() /
      (ecp ? "tests/molecules/Be_pp_conventional_wftrain/wftrain_mdj_ecp.xml" :
             "tests/molecules/H4_ae_conventional_wftrain/wftrain_mdj.xml");
  REQUIRE(definition.parse(fixture.string()));
  xmlNodePtr wavefunction = nullptr;
  for (auto* node = xmlFirstElementChild(definition.getRoot()); node; node = xmlNextElementSibling(node))
  {
    if (xmlStrEqual(node->name, BAD_CAST "particleset")) REQUIRE(pool.put(node));
    if (xmlStrEqual(node->name, BAD_CAST "wavefunction")) wavefunction = node;
  }
  REQUIRE(wavefunction);
  if (csf)
  {
    // Two open-shell CSFs, each with two spin-interchanged determinants. The
    // frozen CSF coefficient is .6, not its first expanded determinant weight.
    Libxml2Document csf_xml;
    REQUIRE(csf_xml.parseFromString(R"(<detlist size="2" type="CSF" nca="0" ncb="0" nea="2" neb="2" nstates="3" cutoff="0">
      <csf id="CI0" coeff="0.6" occ="211" exctLvl="1">
        <det coeff="0.7071067811865475244" alpha="110" beta="101"/>
        <det coeff="0.7071067811865475244" alpha="101" beta="110"/>
      </csf>
      <csf id="CI1" coeff="0" occ="121" exctLvl="1">
        <det coeff="0.7071067811865475244" alpha="110" beta="011"/>
        <det coeff="0.7071067811865475244" alpha="011" beta="110"/>
      </csf></detlist>)"));
    auto* determinantset = xmlFirstElementChild(wavefunction);
    while (determinantset && !xmlStrEqual(determinantset->name, BAD_CAST "determinantset"))
      determinantset = xmlNextElementSibling(determinantset);
    REQUIRE(determinantset);
    auto* old_list = xmlFirstElementChild(xmlFirstElementChild(determinantset));
    REQUIRE(old_list);
    auto* removed = xmlReplaceNode(old_list, xmlDocCopyNode(csf_xml.getRoot(), definition.getRoot()->doc, 1));
    xmlFreeNode(removed);
  }
  for (auto* j = xmlFirstElementChild(wavefunction); j; j = xmlNextElementSibling(j))
    if (xmlStrEqual(j->name, BAD_CAST "jastrow"))
      for (auto* correlation = xmlFirstElementChild(j); correlation; correlation = xmlNextElementSibling(correlation))
        for (auto* coefficient = xmlFirstElementChild(correlation); coefficient; coefficient = xmlNextElementSibling(coefficient))
          xmlSetProp(coefficient, BAD_CAST "optimize", BAD_CAST "no");
  auto& particles = *pool.getParticleSet("e");
  auto& ions = *pool.getParticleSet("ion0");
  ions.update();
  WaveFunctionFactory factory(particles, pool.getPool(), OHMMS::Controller);
  auto source = factory.buildTWF(wavefunction, RuntimeOptions{});
  REQUIRE(source);
  particles.update();
  source->evaluateLog(particles);
  const auto source_log = source->getLogPsi();
  const auto fixed_coefficients = source->findMSD()[0].get().get_C();
  CHECK(source->findMSD()[0].get().fixedCoefficientReference() == 0.6);
  if (csf)
  {
    REQUIRE(fixed_coefficients.size() == 4);
    CHECK(std::abs(fixed_coefficients[0] - 0.6) > 0.1);
  }
  LegacyWaveFunctionAdapter adapter("ci/family-oracle", *source, adapterOptions());
  REQUIRE(adapter.parameterSchema().parameterCount() == 1);
  REQUIRE(adapter.parameterSchema().blocks()[0].update_group == "legacy_linear");
  ParticleSet private_particles(particles);
  auto evaluator = source->makeClone(private_particles);
  private_particles.update();
  evaluator->findMSD()[0].get().isolateOptimizationState();
  const std::vector<double> sample_shifts{-0.12, -0.03, 0.07, 0.18};
  auto replay = std::make_shared<ShiftedLegacyReplay>(*evaluator, private_particles, sample_shifts);
  QMCHamiltonian ham;
  LocalECPotential* local = nullptr;
  StdRandom<QMCTraits::FullPrecRealType> rng(1931);
  ResourceCollection pr("linear_ecp_P"), wr("linear_ecp_W"), hr("linear_ecp_H");
  RefVectorWithLeader<ParticleSet> ps(private_particles, {private_particles});
  RefVectorWithLeader<TrialWaveFunction> ws(*evaluator, {*evaluator});
  RefVectorWithLeader<QMCHamiltonian> hs(ham, {ham});
  const auto terms = localEnergyTermBit(LocalEnergyTerm::KINETIC) |
      (ecp ? localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP) : 0);
  if (ecp)
  {
    evaluator->findMSD()[0].get().enableAcceptedReferenceVirtualRatios();
    ECPComponentBuilder builder("Be", OHMMS::Controller, 2, 1);
    REQUIRE(builder.read_pp_file((fixture.parent_path().parent_path().parent_path() /
        "pseudopotentials_for_tests/Be.BFD.xml").string()));
    auto nonlocal = std::make_unique<NonLocalECPotential>(ions, private_particles, false, true);
    nonlocal->setRandomGenerator(&rng);
    nonlocal->addComponent(0, std::move(builder.pp_nonloc));
    auto local_owner = std::make_unique<LocalECPotential>(ions, private_particles);
    local = local_owner.get();
    local->add(0, std::move(builder.pp_loc), builder.Zeff);
    ham.addOperator(std::make_unique<BareKineticEnergy>(private_particles), "Kinetic");
    ham.addOperator(std::make_unique<CoulombPotential>(private_particles, true, false), "ElecElec");
    ham.addOperator(std::move(local_owner), "LocalECP");
    ham.addOperator(std::move(nonlocal), "NonLocalECP");
    ham.addObservables(private_particles);
    private_particles.createResource(pr); evaluator->createResource(wr); ham.createResource(hr);
  }
  const auto coulomb = [&](const ParticleSet& p) {
    double energy = 0;
    for (int i = 0; i < p.getTotalNum(); ++i)
    {
      for (int j = 0; j < i; ++j) energy += 1.0 / std::sqrt(dot(p.R[i] - p.R[j], p.R[i] - p.R[j]));
      for (const auto& ion : ions.R) energy -= 1.0 / std::sqrt(dot(p.R[i] - ion, p.R[i] - ion));
    }
    for (int i = 0; i < ions.getTotalNum(); ++i)
      for (int j = 0; j < i; ++j) energy += 1.0 / std::sqrt(dot(ions.R[i] - ions.R[j], ions.R[i] - ions.R[j]));
    return energy;
  };
  for (double coefficient : {0.0, 0.1})
  {
    CAPTURE(coefficient);
    auto parameters = adapter.snapshotParameters();
    parameters.values[0] = coefficient;
    const auto frame = std::make_shared<const StructuredParameterSnapshot>(parameters);
    auto op = adapter.makeDerivativeOperator(replay, frame, 0, 0);
    std::vector<std::unique_ptr<StreamingDerivativeOperator>> ecp_operators;
    std::vector<ScoreCovarianceBatch> batches;
    const auto& schema = adapter.parameterSchema();
    std::array<double, 4> score{}, del{}, log{}, phase{}, weights{1, 1, 1, 1};
    std::array<DerivativeValue, 4> energies{};
    auto active = adapter.registrationVariables();
    active[0] = coefficient;
    for (std::size_t sample = 0; sample < 4; ++sample)
    {
      replay->prepareSample(sample);
      adapter.installSnapshot(*evaluator, private_particles, parameters);
      log[sample] = evaluator->getLogPsi();
      phase[sample] = evaluator->getPhase();
      REQUIRE(std::isfinite(log[sample]));
      energies[sample] = kineticEnergy(private_particles) + coulomb(private_particles);
      Vector<QMCTraits::ValueType> scores(1), local_energy(1);
      scores = local_energy = 0;
      evaluator->evaluateDerivatives(private_particles, active, scores, local_energy);
      score[sample] = scores[0];
      del[sample] = local_energy[0];
      if (ecp)
      {
        const auto accepted_positions = private_particles.R;
        auto one = std::make_shared<ShiftedLegacyReplay>(*evaluator, private_particles,
            std::vector<double>{sample_shifts[sample]});
        auto products = adapter.makeDerivativeOperator(one, frame, sample + 1, sample);
        auto consumer = adapter.makeNonLocalECPDerivativeConsumer(one, frame, sample + 1, sample);
        for (int electron = 0; electron < private_particles.getTotalNum(); ++electron)
          CHECK(private_particles.R[electron] == accepted_positions[electron]);
        const std::vector<DerivativeValue> unit{1};
        const VJPCoefficientChannel capture{"capture", DerivativeProduct::LOCAL_ENERGY_VJP,
            coefficientsFor(*products, unit), localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)};
        RecordingParameterSink idle;
        {
          ResourceCollectionTeamLock<ParticleSet> pl(pr, ps);
          ResourceCollectionTeamLock<TrialWaveFunction> wl(wr, ws);
          ResourceCollectionTeamLock<QMCHamiltonian> hl(hr, hs);
          energies[sample] = QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(
              hs, ws, ps, *consumer, {&capture, 1}, idle).local_energies[0];
        }
        CHECK(evaluator->getLogPsi() == log[sample]);
        auto grid = consumer->releaseJVPReplay();
        const auto points = std::vector<NonLocalECPReplayPoint>(grid->points().begin(), grid->points().end());
        products->bindNonLocalECPJVPReplay(std::move(grid));
        const auto positions = private_particles.R;
        const double local_value = local->evaluate(private_particles);
        // Independent signed scalar wavefunction ratios on the same fixed grid.
        const auto fixed_energy = [&](const StructuredParameterSnapshot& point) {
          ParticleSet p(private_particles);
          p.R = positions; p.update();
          auto wf = source->makeClone(p);
          wf->findMSD()[0].get().isolateOptimizationState();
          adapter.installSnapshot(*wf, p, point);
          const double base_log = wf->getLogPsi(), base_phase = wf->getPhase();
          const auto separation = positions[0] - positions[1];
          DerivativeValue energy = kineticEnergy(p) + local_value + 1.0 / std::sqrt(dot(separation, separation));
          for (const auto& knot : points)
          {
            p.R = positions;
            for (int d = 0; d < 3; ++d) p.R[knot.electron][d] = knot.replacement_position[d];
            p.update(); wf->evaluateLog(p);
            energy += knot.bare_weight * std::exp(wf->getLogPsi() - base_log) * std::cos(wf->getPhase() - base_phase);
          }
          const double ratio = std::exp(base_log - log[sample]) * std::cos(base_phase - phase[sample]);
          return std::make_pair(energy, ratio);
        };
        auto plus = parameters, minus = parameters;
        // Psi and H*Psi are exactly linear in this coordinate. A finite signed
        // secant avoids the truncation/cancellation of differentiating E_L alone.
        constexpr double step = 0.125;
        plus.values[0] += step; minus.values[0] -= step;
        const auto positive = fixed_energy(plus), negative = fixed_energy(minus);
        const double physical_score = (positive.second - negative.second) / (2 * step);
        CHECK(physical_score == Catch::Approx(score[sample]).epsilon(2e-12).margin(2e-12));
        score[sample] = physical_score;
        del[sample] = ((positive.first * positive.second - negative.first * negative.second) / (2 * step)).real() -
            energies[sample].real() * physical_score;
        CHECK(fixed_energy(parameters).first.real() == Catch::Approx(energies[sample].real()).epsilon(2e-12).margin(2e-12));
        const VJPCoefficientChannel response{"response", DerivativeProduct::LOCAL_ENERGY_VJP,
            coefficientsFor(*products, unit), terms};
        RecordingParameterSink rows;
        products->applyVJPs({&response, 1}, DerivativeAdjoint::TRANSPOSE, rows);
        CHECK(rows.result()[0][0].real() == Catch::Approx(del[sample]).epsilon(3e-6).margin(3e-8));
        batches.push_back({products.get(), {&weights[sample], 1}, {&energies[sample], 1}});
        ecp_operators.push_back(std::move(products));
      }
      if (csf)
      {
        const double epsilon = 1e-5;
        auto plus = parameters, minus = parameters;
        plus.values[0] += epsilon;
        minus.values[0] -= epsilon;
        adapter.installSnapshot(*evaluator, private_particles, plus);
        const double plus_log = evaluator->getLogPsi();
        const auto plus_energy = kineticEnergy(private_particles);
        adapter.installSnapshot(*evaluator, private_particles, minus);
        CHECK(score[sample] == Catch::Approx((plus_log - evaluator->getLogPsi()) / (2 * epsilon))
            .epsilon(2e-6).margin(2e-8));
        CHECK(del[sample] == Catch::Approx(((plus_energy - kineticEnergy(private_particles)) / (2 * epsilon)).real())
            .epsilon(2e-6).margin(2e-8));
        CHECK(evaluator->findMSD()[0].get().fixedCoefficientReference() == 0.6);
      }
    }
    double mean = 0, mean_energy = 0;
    for (int i = 0; i < 4; ++i) { mean += score[i] / 4; mean_energy += energies[i].real() / 4; }
    std::array<double, 4> dense_s{}, dense_h{}, dense_raw{};
    for (int i = 0; i < 4; ++i)
    {
      const double d[] = {1, score[i] - mean}, l[] = {energies[i].real(), del[i] + energies[i].real() * d[1]};
      for (int row = 0; row < 2; ++row)
        for (int column = 0; column < 2; ++column)
        {
          dense_s[2 * row + column] += d[row] * d[column] / 4;
          dense_h[2 * row + column] += (d[row] * l[column] + l[row] * d[column]) / 8;
          dense_raw[2 * row + column] += d[row] * l[column] / 4;
        }
    }
    REQUIRE(dense_s[3] > 1e-10);
    const bool compare_legacy = !csf && !ecp;
    qmcplusplus::testing::LegacyP1Matrices legacy{};
    if (compare_legacy)
    {
      std::array<qmcplusplus::testing::LegacyP1Sample, 4> records;
      for (int sample = 0; sample < 4; ++sample)
      {
        records[sample] = {weights[sample], energies[sample].real(), score[sample], del[sample]};
        replay->prepareSample(sample);
        // Psi and H*Psi are exactly linear in C1. Independent signed secants
        // validate the physical records shared by old/new matrix assembly.
        const auto physical = [&](double displacement) {
          ParticleSet p(private_particles);
          auto wf = source->makeClone(p);
          wf->findMSD()[0].get().isolateOptimizationState();
          p.update();
          auto point = parameters; point.values[0] += displacement;
          adapter.installSnapshot(*wf, p, point);
          const double ratio = std::exp(wf->getLogPsi() - log[sample]) *
              std::cos(wf->getPhase() - phase[sample]);
          return std::array<double, 2>{ratio, (kineticEnergy(p).real() + coulomb(p)) * ratio};
        };
        const auto plus = physical(.125), minus = physical(-.125);
        CHECK((plus[0] - minus[0]) / .25 == Catch::Approx(score[sample]).epsilon(2e-10).margin(2e-10));
        CHECK((plus[1] - minus[1]) / .25 - energies[sample].real() * score[sample] ==
              Catch::Approx(del[sample]).epsilon(2e-10).margin(2e-10));
      }
      legacy = qmcplusplus::testing::legacyP1Matrices(records);
      // Old fill uses RAW centered H. New LM symmetrizes it; neither policy
      // implies equal updates. Old all-LINEAR rescale=1 omits the fixed-C0
      // projective denominator already checked physically below for new LM.
      CHECK(legacy.linear_rescale == 1.0);
      for (int entry = 0; entry < 4; ++entry)
      {
        CHECK(legacy.hamiltonian[entry] == Catch::Approx(dense_raw[entry]).epsilon(2e-10).margin(2e-10));
        CHECK(legacy.overlap[entry] == Catch::Approx(dense_s[entry]).epsilon(2e-10).margin(2e-10));
      }
    }
    if (!ecp) batches.push_back({op.get(), {weights.data(), 4}, {energies.data(), 4}});
    const DerivativeArrayView<const ScoreCovarianceBatch> views{batches.data(), batches.size()};
    ScoreCovarianceFactor factor(schema, parameters.version, views);
    auto rng_before_actions = rng;
    SampledLinearMethodOperator h(factor, views, terms, false, true, 0);
    SampledLinearMethodOperator s(factor, views, terms, true, false, 0);
    SampledLinearMethodOperator raw_h(factor, views, terms, false, false, 0);
    for (int column = 0; column < 2; ++column)
    {
      std::array<DerivativeValue, 2> direction{}, h_action{}, s_action{}, raw_action{};
      direction[column] = 1;
      h.apply({direction.data(), 2}, {h_action.data(), 2});
      s.apply({direction.data(), 2}, {s_action.data(), 2});
      raw_h.apply({direction.data(), 2}, {raw_action.data(), 2});
      for (int row = 0; row < 2; ++row)
      {
        CHECK(h_action[row].real() == Catch::Approx(dense_h[2 * row + column]).epsilon(2e-10).margin(2e-10));
        CHECK(s_action[row].real() == Catch::Approx(dense_s[2 * row + column]).epsilon(2e-10).margin(2e-10));
        CHECK(raw_action[row].real() == Catch::Approx(dense_raw[2 * row + column]).epsilon(2e-10).margin(2e-10));
        if (compare_legacy)
        {
          CHECK(raw_action[row].real() == Catch::Approx(legacy.hamiltonian[2 * row + column]).epsilon(2e-10).margin(2e-10));
          CHECK(s_action[row].real() == Catch::Approx(legacy.overlap[2 * row + column]).epsilon(2e-10).margin(2e-10));
          CHECK(h_action[row].real() == Catch::Approx(0.5 * (legacy.hamiltonian[2 * row + column] + legacy.hamiltonian[2 * column + row]))
                .epsilon(2e-10).margin(2e-10));
        }
      }
    }
    MatrixFreeLinearMethodControl lm;
    lm.maximum_subspace_dimension = 2;
    lm.maximum_iterations = 16;
    lm.maximum_operator_applications = 64;
    lm.maximum_seconds = 5;
    const std::array<DerivativeValue, 2> initial{1, 0};
    IdentityPreconditioner preconditioner(schema, parameters.version);
    const auto result = solveSymmetrizedMatrixFreeLinearMethod(h, s, preconditioner, {initial.data(), 2}, lm);
    INFO(linearMethodStopReasonName(result.stop_reason));
    REQUIRE(result.converged);
    AcceleratedLinearMethodControl alm;
    alm.outer = lm;
    const auto accelerated = solveAcceleratedMatrixFreeLinearMethod(raw_h, s, preconditioner,
        {initial.data(), 2}, alm);
    INFO(linearMethodStopReasonName(accelerated.stop_reason));
    // S is diag(1, variance) in the centered P1 basis. Preserve a genuine
    // complex-conjugate raw pencil instead of tuning samples to force a root.
    const double trace = dense_raw[0] + dense_raw[3] / dense_s[3];
    const double determinant = (dense_raw[0] * dense_raw[3] - dense_raw[1] * dense_raw[2]) / dense_s[3];
    CAPTURE(dense_raw, dense_s, trace, determinant);
    if (ecp && trace * trace - 4 * determinant < 0)
    {
      CHECK_FALSE(accelerated.converged);
      CHECK(accelerated.stop_reason == LinearMethodStopReason::UNTRUSTED_ROOT);
    }
    else
      REQUIRE(accelerated.converged);
    const auto gradient = factor.scoreCovarianceGradient();
    const ParameterGradientView objective{schema.fingerprint(), parameters.version, ReductionDomain::GLOBAL,
                                         {gradient.data(), gradient.size()}};
    for (const auto& solved : {result, accelerated})
    {
      if (!solved.converged) continue; // Only the independently proved complex-root ECP case above.
      for (double rate : {1.0, 0.25})
        for (double xi : {0.0, 0.8})
        {
          const double v0 = solved.eigenvector[0].real(), v1 = solved.eigenvector[1].real();
          const double gauge = v0 - mean * v1;
          REQUIRE(std::abs(gauge) > 1e-10);
          LinearMethodUpdateRule update(factor, solved, rate, xi,
              LinearMethodNormalization::FIXED_REFERENCE_LINEAR, views);
          const auto mapped = update.propose(schema, parameters, objective);
          CHECK(mapped.values[0] == Catch::Approx(coefficient + rate * v1 / gauge).epsilon(2e-10).margin(2e-10));
          for (std::size_t sample = 0; sample < 4; ++sample)
          {
            replay->prepareSample(sample);
            adapter.installSnapshot(*evaluator, private_particles, mapped);
            const double ratio = std::exp(evaluator->getLogPsi() - log[sample]) *
                std::cos(evaluator->getPhase() - phase[sample]);
            CHECK(ratio == Catch::Approx(1 + rate * ((v0 + v1 * (score[sample] - mean)) / gauge - 1))
                .epsilon(2e-10).margin(2e-10));
            CHECK(evaluator->findMSD()[0].get().get_C()[0] == fixed_coefficients[0]);
            CHECK(evaluator->findMSD()[0].get().fixedCoefficientReference() == 0.6);
            if (csf)
              CHECK(evaluator->findMSD()[0].get().get_C()[1] == fixed_coefficients[1]);
          }
        }
    }
    ProjectedInverseIterationFactor pii_factor(factor, terms);
    PIIUpdateControl control;
    control.shift_mode = PIIShiftMode::EXPLICIT;
    control.explicit_shift = mean_energy - 1;
    control.damping = 0.3;
    control.learning_rate = 0.01;
    control.maximum_damping_retries = 0;
    control.gmres.restart_length = 4;
    control.gmres.krylov.maximum_iterations = 16;
    control.gmres.krylov.maximum_operator_applications = 64;
    control.gmres.krylov.maximum_seconds = 5;
    double rhs = 0, denominator = control.damping;
    for (int i = 0; i < 4; ++i)
    {
      const double centered = score[i] - mean;
      rhs += centered * (energies[i].real() - mean_energy) / 4;
      denominator += centered * (del[i] + (energies[i].real() - control.explicit_shift) * centered) / 4;
    }
    REQUIRE(std::abs(denominator) > 1e-10);
    ProjectedInverseIterationUpdateRule pii(pii_factor, control);
    MinimumProjectedInverseIterationUpdateRule minpii(pii_factor, control);
    const double expected = coefficient - control.learning_rate * rhs / denominator;
    for (const auto& mapped : {pii.propose(schema, parameters, objective),
                              minpii.propose(schema, parameters, objective)})
    {
      CHECK(mapped.values[0] == Catch::Approx(expected).epsilon(2e-9).margin(2e-11));
      // Independent linear CSF coordinates use the additive physical map, not
      // LM's centered Ritz-vector gauge. Test the actual expanded wavefunction.
      for (std::size_t sample = 0; sample < 4; ++sample)
      {
        replay->prepareSample(sample);
        adapter.installSnapshot(*evaluator, private_particles, mapped);
        const double ratio = std::exp(evaluator->getLogPsi() - log[sample]) *
            std::cos(evaluator->getPhase() - phase[sample]);
        CHECK(ratio == Catch::Approx(1 + (mapped.values[0] - coefficient) * score[sample])
            .epsilon(2e-10).margin(2e-10));
        const auto& md = evaluator->findMSD()[0].get();
        CHECK(md.fixedCoefficientReference() == 0.6);
        CHECK(md.get_C()[0] == fixed_coefficients[0]);
        if (csf)
        {
          CHECK(md.get_C()[1] == fixed_coefficients[1]);
          CHECK(md.get_C()[2] == Catch::Approx(mapped.values[0] * std::sqrt(0.5)).epsilon(2e-15));
          CHECK(md.get_C()[3] == md.get_C()[2]);
        }
      }
    }
    pii.proposalRejected();
    minpii.proposalRejected();
    source->evaluateLog(particles);
    CHECK(source->getLogPsi() == source_log);
    CHECK(source->findMSD()[0].get().get_C() == fixed_coefficients);
    CHECK(adapter.snapshotParameters().values[0] == (ecp ? 0.05 : 0));
    if (ecp) CHECK(rng() == rng_before_actions());
  }
}

TEST_CASE("two-occupied orbital ratio tangents use only the accepted inverse",
          "[legacy_adapter][two_occupied_ratio]")
{
  ParticleSetPool pool(OHMMS::Controller);
  Libxml2Document particle_xml, definition;
  REQUIRE(particle_xml.parseFromString(R"(<qmcsystem>
    <particleset name="ion0" size="1"><group name="Be"><parameter name="charge">2</parameter>
      </group><attrib name="position" datatype="posArray">0 0 0</attrib></particleset>
    <particleset name="e"><group name="u" size="2"><parameter name="charge">-1</parameter>
      <parameter name="mass">1</parameter><attrib name="position" datatype="posArray">
      .3 .2 .4  -.5 .3 .7</attrib></group>
      <group name="d" size="2"><parameter name="charge">-1</parameter><parameter name="mass">1</parameter>
      <attrib name="position" datatype="posArray">.6 -.2 .3  -.4 -.3 .8</attrib></group>
    </particleset></qmcsystem>)"));
  REQUIRE(pool.put(xmlFirstElementChild(particle_xml.getRoot())));
  REQUIRE(pool.put(xmlNextElementSibling(xmlFirstElementChild(particle_xml.getRoot()))));
  REQUIRE(definition.parseFromString(R"(<wavefunction name="psi0" target="e">
    <sposet_collection type="MolecularOrbital" source="ion0">
      <basisset keyword="STO" transform="no"><atomicBasisSet type="STO" elementType="Be"
        angular="cartesian" normalized="no">
        <basisGroup rid="S" n="1" l="0" m="0" type="Slater"><radfunc n="1" exponent="1.2"/></basisGroup>
        <basisGroup rid="P" n="2" l="1" type="Slater"><radfunc n="2" exponent="1.2"/></basisGroup>
      </atomicBasisSet></basisset>
      <rotated_sposet name="rot"><sposet name="base" size="4"><coefficient size="4" id="C">
        1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1
      </coefficient></sposet></rotated_sposet>
    </sposet_collection><determinantset><slaterdeterminant>
      <determinant sposet="rot"/><determinant sposet="rot"/>
    </slaterdeterminant></determinantset></wavefunction>)"));
  auto& particles = *pool.getParticleSet("e");
  pool.getParticleSet("ion0")->update();
  WaveFunctionFactory factory(particles, pool.getPool(), OHMMS::Controller);
  auto source = factory.buildTWF(definition.getRoot(), RuntimeOptions{});
  REQUIRE(source);
  particles.update();
  source->evaluateLog(particles);
  auto& slater = source->findSD().front().get();
  auto& rotated = dynamic_cast<RotatedSPOs&>(slater.getPhi(0));
  REQUIRE(&slater.getPhi(0) == &slater.getPhi(1));
  const auto source_coefficients = *dynamic_cast<LCAOrbitalSet&>(*rotated.Phi_).C;
  const double source_log = source->getLogPsi();
  LegacyWaveFunctionAdapter adapter("conventional", *source, adapterOptions());
  const auto active = adapter.registrationVariables();
  REQUIRE(active.size_of_active() == 4);
  const auto initial = adapter.snapshotParameters();
  using Matrix = SPOSet::ValueMatrix;
  using Vector = SPOSet::ValueVector;
  // Independent determinant: no production matrix inverse or ratio helper.
  const auto determinant = [](const Matrix& m) { return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0); };
  for (int spin = 0; spin < 2; ++spin)
  {
    auto& det = dynamic_cast<DiracDeterminant<>&>(slater.getDet(spin));
    const int first = det.getFirstIndex();
    Matrix accepted(2, 4), laplacians(2, 4);
    SPOSet::GradMatrix gradients(2, 4);
    rotated.Phi_->evaluate_notranspose(particles, first, first + 2, accepted, gradients, laplacians);
    const Matrix inverse = det.getAcceptedInverseTranspose(), original = accepted;
    for (int electron = 0; electron < 2; ++electron)
    {
      Vector replacement(4), output(4);
      // Use a fresh private point for a full LCAO coefficient-rotation FD oracle.
      ParticleSet moved(particles);
      moved.R[first + electron] = {.1, -.6, .9}; moved.update();
      auto moved_wf = source->makeClone(moved);
      auto& moved_rot = dynamic_cast<RotatedSPOs&>(moved_wf->findSD()[0].get().getPhi(spin));
      moved_rot.Phi_->evaluateValue(moved, first + electron, replacement);
      output = 0;
      const auto ratio = rotated.evaluateAcceptedReferenceOVRatioDerivatives(accepted, inverse, replacement, electron, active, output);
      Matrix virtual_matrix = accepted;
      for (int q = 0; q < 4; ++q) virtual_matrix(electron, q) = replacement[q];
      CHECK(ratio == Catch::Approx(determinant(virtual_matrix) / determinant(accepted)).epsilon(2e-12).margin(2e-12));
      for (int parameter = 0; parameter < 4; ++parameter)
      {
        const auto physical_ratio = [&](double displacement) {
          ParticleSet p(particles);
          auto wf = source->makeClone(p);
          // Normal orbital clones share coefficients; detach this fresh FD bank
          // explicitly. Both spin determinants share this one rotated wrapper.
          dynamic_cast<RotatedSPOs&>(wf->findSD()[0].get().getPhi(0)).isolateOptimizationState();
          p.update();
          auto point = initial; point.values[parameter] += displacement;
          adapter.installSnapshot(*wf, p, point);
          const double base_log = wf->getLogPsi(), base_phase = wf->getPhase();
          p.R[first + electron] = moved.R[first + electron]; p.update(); wf->evaluateLog(p);
          return std::exp(wf->getLogPsi() - base_log) * std::cos(wf->getPhase() - base_phase);
        };
        CHECK(output[parameter] == Catch::Approx((physical_ratio(1e-5) - physical_ratio(-1e-5)) / 2e-5)
              .epsilon(3e-6).margin(3e-8));
      }
    }
    for (std::size_t i = 0; i < accepted.size(); ++i) CHECK(accepted.data()[i] == original.data()[i]);
    for (std::size_t i = 0; i < inverse.size(); ++i) CHECK(inverse.data()[i] == det.getAcceptedInverseTranspose().data()[i]);
  }
  // Map local active deltas after noncommuting history, not full-log coordinates.
  {
    auto owner = rotated.makeClone();
    auto& accepted_rot = dynamic_cast<RotatedSPOs&>(*owner);
    accepted_rot.isolateOptimizationState();
    auto history = active;
    history[0] = .17; accepted_rot.resetParametersExclusive(history);
    history[1] = -.11; accepted_rot.resetParametersExclusive(history);
    const auto full = accepted_rot.fullRotationCoordinates();
    REQUIRE(full[5] != 0); // generated virtual/virtual component
    auto& leaf = dynamic_cast<LCAOrbitalSet&>(*accepted_rot.Phi_);
    const auto coefficients = *leaf.C, reference = leaf.originalRotationCoefficients();
    auto fixed = accepted_rot.Phi_->makeClone();
    dynamic_cast<LCAOrbitalSet&>(*fixed).isolateOptimizationState();
    CHECK(dynamic_cast<LCAOrbitalSet&>(*fixed).C != leaf.C);
    MultiDiracDeterminant block(std::move(fixed), false, 0, 2);
    std::vector<ci_configuration2> configurations(4);
    configurations[0].occup = {0, 1}; configurations[1].occup = {0, 2};
    configurations[2].occup = {1, 2}; configurations[3].occup = {2, 3};
    std::vector<std::size_t> mapping;
    block.createDetData(0, configurations, {0, 1, 2, 3}, mapping);
    block.evaluateForWalkerMove(particles);
    Matrix base(2, 4), laps(2, 4);
    SPOSet::GradMatrix gradients(2, 4);
    accepted_rot.evaluate_notranspose(particles, 0, 2, base, gradients, laps);
    const auto denominator = determinant(base);
    REQUIRE(std::abs(denominator) > 1e-8);
    ParticleSet moved(particles); moved.R[0] = {.1, -.6, .9}; moved.update();
    Vector ratios(4), generated(4);
    for (int parameter = 0; parameter < 4; ++parameter)
    {
      const int p = parameter / 2, q = 2 + parameter % 2;
      block.evaluateOrbitalGeneratorToAcceptedReference(moved, 0, 0, p, q, ratios, generated);
      const auto dense = [&](double delta) {
        auto probe = accepted_rot.makeClone();
        auto& rotation = dynamic_cast<RotatedSPOs&>(*probe);
        rotation.isolateOptimizationState(); // fresh bank/frame for EACH +/- probe
        auto point = history; point[parameter] += delta; rotation.resetParametersExclusive(point);
        Matrix values(2, 4);
        rotation.evaluate_notranspose(moved, 0, 2, values, gradients, laps);
        std::array<QMCTraits::ValueType, 4> result;
        for (int term = 0; term < 4; ++term)
        {
          const auto& occ = block.getConfiguration(term).occup;
          result[term] = (values(0, occ[0]) * values(1, occ[1]) -
                          values(0, occ[1]) * values(1, occ[0])) / denominator;
        }
        return result;
      };
      const auto plus = dense(1e-5), minus = dense(-1e-5), zero = dense(0);
      for (int term = 0; term < 4; ++term)
      {
        CHECK(std::real(ratios[term]) == Catch::Approx(std::real(zero[term])).margin(2e-12));
        CHECK(std::real(generated[term]) == Catch::Approx(std::real((plus[term] - minus[term]) / 2e-5)).epsilon(3e-6).margin(3e-8));
      }
    }
    CHECK(accepted_rot.fullRotationCoordinates() == full);
    optimize::VariableSet after; accepted_rot.checkInVariablesExclusive(after);
    for (int i = 0; i < 4; ++i) CHECK(after[i] == history[i]);
    for (std::size_t i = 0; i < coefficients.size(); ++i)
    { CHECK(leaf.C->data()[i] == coefficients.data()[i]); CHECK(leaf.originalRotationCoefficients().data()[i] == reference.data()[i]); }
    MultiDiracDeterminant rejected(accepted_rot.makeClone(), false, 0, 2);
    rejected.createDetData(0, configurations, {0, 1, 2, 3}, mapping); rejected.evaluateForWalkerMove(particles);
    REQUIRE(rejected.getPhi()->isOptimizable());
    CHECK_THROWS_AS(rejected.evaluateOrbitalGeneratorToAcceptedReference(moved, 0, 0, 0, 2, ratios, generated), std::invalid_argument);
  }
  // Binary-exact, signed nonsymmetric reference makes orientation errors visible.
  Matrix accepted(2, 4), inverse(2, 2);
  const double entries[]{1, 2, 3, 5, 0, -2, 7, 11};
  std::copy(std::begin(entries), std::end(entries), accepted.data());
  inverse(0, 0) = 1; inverse(0, 1) = 0;
  inverse(1, 0) = 1; inverse(1, 1) = -.5;
  Vector replacement(4), output(4);
  replacement[0] = 0; replacement[1] = -2; replacement[2] = 13; replacement[3] = 17;
  output = 0;
  CHECK(rotated.evaluateAcceptedReferenceOVRatioDerivatives(accepted, inverse, replacement, 0, active, output) == 0);
  CHECK(output[0] == 6); CHECK(output[1] == 6);
  CHECK(output[2] == 0); CHECK(output[3] == 0);
  for (int parameter = 0; parameter < 4; ++parameter)
  {
    const int occupied = parameter / 2, virtual_orbital = 2 + parameter % 2;
    const auto dense_ratio = [&](double angle) {
      Matrix a = accepted, q = accepted;
      for (int row = 0; row < 2; ++row)
        a(row, occupied) = std::cos(angle) * accepted(row, occupied) +
            std::sin(angle) * accepted(row, virtual_orbital);
      q = a;
      for (int orbital = 0; orbital < 4; ++orbital) q(0, orbital) = replacement[orbital];
      q(0, occupied) = std::cos(angle) * replacement[occupied] +
          std::sin(angle) * replacement[virtual_orbital];
      return determinant(q) / determinant(a);
    };
    CHECK(output[parameter] == Catch::Approx((dense_ratio(1e-6) - dense_ratio(-1e-6)) / 2e-6)
          .epsilon(2e-9).margin(2e-9));
  }
  for (int electron = 0; electron < 2; ++electron)
  {
    for (int q = 0; q < 4; ++q) replacement[q] = accepted(electron, q);
    output = 0;
    CHECK(rotated.evaluateAcceptedReferenceOVRatioDerivatives(accepted, inverse, replacement, electron, active, output) == 1);
    for (int parameter = 0; parameter < 4; ++parameter) CHECK(output[parameter] == 0);
  }
  CHECK_THROWS(rotated.evaluateAcceptedReferenceOVRatioDerivatives(accepted, inverse, replacement, 2, active, output));
  replacement[3] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(rotated.evaluateAcceptedReferenceOVRatioDerivatives(accepted, inverse, replacement, 0, active, output));
  replacement[3] = 17;
  accepted(0, 0) = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(rotated.evaluateAcceptedReferenceOVRatioDerivatives(accepted, inverse, replacement, 0, active, output));
  accepted(0, 0) = 1;
  inverse(0, 0) = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(rotated.evaluateAcceptedReferenceOVRatioDerivatives(accepted, inverse, replacement, 0, active, output));
  // Independent signed 3x3 oracle: accepted inverse is analytic, while virtual
  // matrices are evaluated by the explicit determinant polynomial, never inverted.
  RotatedSPOs three("three", rotated.Phi_->makeClone());
  three.buildOptVariables(3);
  optimize::VariableSet three_active;
  three.checkInVariablesExclusive(three_active); three_active.resetIndex(); three.checkOutVariables(three_active);
  Matrix a3(3, 4), b3(3, 3);
  const double a_values[]{1, 2, 0, 3, 0, 1, 1, 5, 0, 0, -2, 7};
  const double b_values[]{1, 0, 0, -2, 1, 0, -1, .5, -.5};
  std::copy(std::begin(a_values), std::end(a_values), a3.data());
  std::copy(std::begin(b_values), std::end(b_values), b3.data());
  const auto det3 = [](const Matrix& a) {
    return a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) -
        a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) +
        a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
  };
  Vector d3(3);
  for (int electron = 0; electron < 3; ++electron)
  {
    for (int orbital = 0; orbital < 4; ++orbital) replacement[orbital] = a3((electron + 1) % 3, orbital);
    replacement[3] = 13; d3 = 0;
    CHECK(three.evaluateAcceptedReferenceOVRatioDerivatives(a3, b3, replacement, electron, three_active, d3) == 0);
    for (int occupied = 0; occupied < 3; ++occupied)
    {
      const auto ratio3 = [&](double angle) {
        Matrix a = a3, q = a3;
        for (int row = 0; row < 3; ++row)
          a(row, occupied) = std::cos(angle) * a3(row, occupied) + std::sin(angle) * a3(row, 3);
        q = a;
        for (int orbital = 0; orbital < 4; ++orbital) q(electron, orbital) = replacement[orbital];
        q(electron, occupied) = std::cos(angle) * replacement[occupied] + std::sin(angle) * replacement[3];
        return det3(q) / det3(a);
      };
      CHECK(d3[occupied] == Catch::Approx((ratio3(1e-6) - ratio3(-1e-6)) / 2e-6).epsilon(2e-9).margin(2e-9));
    }
  }
  const auto& current_coefficients = *dynamic_cast<LCAOrbitalSet&>(*rotated.Phi_).C;
  for (std::size_t i = 0; i < source_coefficients.size(); ++i)
    CHECK(current_coefficients.data()[i] == source_coefficients.data()[i]);
  source->evaluateLog(particles);
  CHECK(source->getLogPsi() == source_log);
}

TEST_CASE("accepted-reference orbital ECP products are nodal safe", "[legacy_adapter][orbital_ecp]")
{
  // The nodal model uses the supported octahedral BFD rule and a counting
  // identity-rotation RNG. Pure px vanishes on an exact q_x=0 symmetry axis.
  const std::string node_coefficients = "0 1 0 0  1 0 0 0  0 0 1 0";
  for (int model = 0; model < 10; ++model)
  {
    CAPTURE(model);
    const bool joint = model >= 5;
    const bool polarized = model == 7 || model == 8;
    const bool csf = model == 9;
    const bool md = model >= 3, node = model == 1 || model == 4 || model == 6;
    const bool many = model >= 2;
    ParticleSetPool pool(OHMMS::Controller);
    Libxml2Document particle_xml;
    REQUIRE(particle_xml.parseFromString(R"(<qmcsystem>
      <particleset name="ion0" size="1"><group name="Be">
        <parameter name="charge">2</parameter><parameter name="atomicnumber">4</parameter>
        </group><attrib name="position" datatype="posArray">0 0 0</attrib></particleset>
      <particleset name="e"><group name="u" size="1">
        <parameter name="charge">-1</parameter><parameter name="mass">1</parameter>
        <attrib name="position" datatype="posArray">0.2 -0.3 0.4</attrib></group>
        <group name="d" size="1"><parameter name="charge">-1</parameter><parameter name="mass">1</parameter>
        <attrib name="position" datatype="posArray">-0.5 0.1 0.7</attrib></group></particleset>
      </qmcsystem>)"));
    if (many)
    {
      auto* ion = xmlFirstElementChild(particle_xml.getRoot());
      auto* species = xmlFirstElementChild(ion);
      xmlSetProp(species, BAD_CAST "name", BAD_CAST "C");
      xmlNodeSetContent(xmlFirstElementChild(species), BAD_CAST "4");
      xmlNodeSetContent(xmlNextElementSibling(xmlFirstElementChild(species)), BAD_CAST "6");
      auto* spin = xmlFirstElementChild(xmlNextElementSibling(ion));
      for (const char* positions : {".2 -.3 .4  .6 .2 -.5", "-.5 .1 .7  -.8 -.4 -.2"})
      {
        xmlSetProp(spin, BAD_CAST "size", BAD_CAST "2");
        for (auto* child = xmlFirstElementChild(spin); child; child = xmlNextElementSibling(child))
          if (xmlStrEqual(child->name, BAD_CAST "attrib")) xmlNodeSetContent(child, BAD_CAST positions);
        spin = xmlNextElementSibling(spin);
      }
    }
    if (polarized)
    {
      // C^- is charged: five electrons, not the neutral four-electron case.
      INFO("C^- unequal-spin model; three-electron group=" << (model == 7 ? "u" : "d"));
      auto* spin = xmlFirstElementChild(xmlNextElementSibling(xmlFirstElementChild(particle_xml.getRoot())));
      if (model == 8) spin = xmlNextElementSibling(spin);
      xmlSetProp(spin, BAD_CAST "size", BAD_CAST "3");
      for (auto* child = xmlFirstElementChild(spin); child; child = xmlNextElementSibling(child))
        if (xmlStrEqual(child->name, BAD_CAST "attrib"))
          xmlNodeSetContent(child, BAD_CAST (model == 7 ? ".2 -.3 .4  .6 .2 -.5  .4 -.7 .6" :
                                                         "-.5 .1 .7  -.8 -.4 -.2  .4 -.7 .6"));
    }
    REQUIRE(pool.put(xmlFirstElementChild(particle_xml.getRoot())));
    REQUIRE(pool.put(xmlNextElementSibling(xmlFirstElementChild(particle_xml.getRoot()))));
    auto& ions = *pool.getParticleSet("ion0");
    auto& particles = *pool.getParticleSet("e");
    ions.update();
    if (polarized)
    {
      REQUIRE(particles.groupsize(0) == (model == 7 ? 3 : 2));
      REQUIRE(particles.groupsize(1) == (model == 7 ? 2 : 3));
    }
    if (model == 1)
    {
      particles.R[0] = {1, 0, 0};
      particles.R[1] = {-2, 0, 0};
    }
    if (model == 4 || model == 6)
    {
      particles.R[0] = {1, 1, 0}; particles.R[1] = {0, 0, 1};
      particles.R[2] = {-1, 1, 0}; particles.R[3] = {0, 0, 2};
    }
    const std::string coefficients = many ? "1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1" :
        model == 0 ? "1 0 0 0  0 1 0 0  0 0 1 0" : node_coefficients;
    REQUIRE(!coefficients.empty());
    Libxml2Document definition;
    std::string wavefunction_xml = std::string{R"(<wavefunction name="psi0" target="e">
      <sposet_collection type="MolecularOrbital" source="ion0">
        <basisset keyword="STO" transform="no"><atomicBasisSet type="STO" elementType="Be"
            angular="cartesian" normalized="no">
          <basisGroup rid="S" n="1" l="0" m="0" type="Slater"><radfunc n="1" exponent="1.2"/></basisGroup>
          <basisGroup rid="P" n="2" l="1" type="Slater"><radfunc n="2" exponent="1.2"/></basisGroup>
        </atomicBasisSet></basisset>
        <rotated_sposet name="rot"><sposet name="base" size="3">
          <coefficient size="3" id="accepted">)"} + coefficients + R"(</coefficient>
        </sposet></rotated_sposet>
      </sposet_collection><determinantset><slaterdeterminant>
        <determinant sposet="rot"/><determinant sposet="rot"/>
      </slaterdeterminant></determinantset>
      <jastrow name="J2" type="Two-Body" function="Bspline">
        <correlation rcut="4" size="4" speciesA="u" speciesB="d">
          <coefficients id="ud" type="Array" optimize="yes">0.12 -0.08 0.04 -0.01</coefficients>
        </correlation>
      </jastrow></wavefunction>)";
    if (many)
      for (const auto& [before, after] : std::array<std::pair<std::string, std::string>, 2>{{
          {"elementType=\"Be\"", "elementType=\"C\""}, {"size=\"3\"", "size=\"4\""}}})
        for (std::size_t position = 0; (position = wavefunction_xml.find(before, position)) != std::string::npos; position += after.size())
          wavefunction_xml.replace(position, before.size(), after);
    if (md)
    {
      const auto begin = wavefunction_xml.find("<rotated_sposet");
      const auto end = wavefunction_xml.find("</rotated_sposet>", begin) + std::string("</rotated_sposet>").size();
      auto down = wavefunction_xml.substr(begin, end - begin);
      for (const auto& [before, after] : {std::pair{"name=\"rot\"", "name=\"rot_dn\""},
                                       std::pair{"name=\"base\"", "name=\"base_dn\""}})
        down.replace(down.find(before), std::string(before).size(), after);
      wavefunction_xml.insert(end, down);
      const auto first = wavefunction_xml.find("<slaterdeterminant>"),
                 last = wavefunction_xml.find("</slaterdeterminant>") + std::string("</slaterdeterminant>").size();
      wavefunction_xml.replace(first, last - first, R"(<multideterminant optimize="no" spo_0="rot" spo_1="rot_dn" algorithm="precomputed_table_method">
        <detlist size="2" type="DETS" nc0="0" ne0="2" nc1="0" ne1="2" nstates="4" cutoff="1e-20">
          <ci id="CI0" coeff="1" occ0="1100" occ1="1100"/>
          <ci id="CI1" coeff="0.1" occ0="1010" occ1="1010"/>
        </detlist></multideterminant>)");
      if (joint)
      {
        const auto flag = wavefunction_xml.find("multideterminant optimize=\"no\"");
        REQUIRE(flag != std::string::npos);
        wavefunction_xml.replace(flag, std::string("multideterminant optimize=\"no\"").size(),
                                "multideterminant optimize=\"yes\"");
      }
    }
    REQUIRE(definition.parseFromString(wavefunction_xml));
    if (csf)
    {
      // Deliberately unnormalized frozen CSF basis: actual CI0=.6, expanded C0=.48.
      // Active CI1 is a genuine signed open-shell two-determinant combination.
      Libxml2Document expansion;
      REQUIRE(expansion.parseFromString(R"(<detlist size="2" type="CSF" nca="0" ncb="0"
          nea="2" neb="2" nstates="4" cutoff="0">
        <csf id="CI0" coeff="0.6" occ="2200" exctLvl="0">
          <det coeff="0.8" alpha="1100" beta="1100"/>
        </csf>
        <csf id="CI1" coeff="0.1" occ="2110" exctLvl="1">
          <det coeff="0.7071067811865475244" alpha="1100" beta="1010"/>
          <det coeff="-0.7071067811865475244" alpha="1010" beta="1100"/>
        </csf></detlist>)"));
      auto* determinantset = xmlNextElementSibling(xmlFirstElementChild(definition.getRoot()));
      auto* old_list = xmlFirstElementChild(xmlFirstElementChild(determinantset));
      auto* removed = xmlReplaceNode(old_list, xmlDocCopyNode(expansion.getRoot(), definition.getRoot()->doc, 1));
      xmlFreeNode(removed);
    }
    if (polarized)
    {
      auto* detlist = xmlFirstElementChild(xmlFirstElementChild(
          xmlNextElementSibling(xmlFirstElementChild(definition.getRoot()))));
      REQUIRE(xmlStrEqual(detlist->name, BAD_CAST "detlist"));
      const char* electron_count = model == 7 ? "ne0" : "ne1";
      const char* occupation = model == 7 ? "occ0" : "occ1";
      xmlSetProp(detlist, BAD_CAST electron_count, BAD_CAST "3");
      auto* ci = xmlFirstElementChild(detlist);
      xmlSetProp(ci, BAD_CAST occupation, BAD_CAST "1110");
      xmlSetProp(xmlNextElementSibling(ci), BAD_CAST occupation, BAD_CAST "1101");
    }
    WaveFunctionFactory factory(particles, pool.getPool(), OHMMS::Controller);
    auto source = factory.buildTWF(definition.getRoot(), RuntimeOptions{});
    REQUIRE(source);
    const auto rotated = [md](TrialWaveFunction& wf, int spin = 0) -> RotatedSPOs& {
      if (md) return dynamic_cast<RotatedSPOs&>(*wf.findMSD().front().get().getDet(spin).getPhi());
      auto& slater = wf.findSD().front().get();
      REQUIRE(&slater.getPhi(0) == &slater.getPhi(1));
      return dynamic_cast<RotatedSPOs&>(slater.getPhi(0));
    };
    const auto isolate = [&](TrialWaveFunction& wf) {
      for (int spin = 0; spin < (md ? 2 : 1); ++spin) rotated(wf, spin).isolateOptimizationState();
      if (md) wf.findMSD().front().get().isolateOptimizationState();
    };
    // Cartesian input expands a complete p shell. Old models select three
    // orbitals; neutral carbon retains the full four-orbital identity bank.
    auto& initial_lcao = dynamic_cast<LCAOrbitalSet&>(*rotated(*source).Phi_);
    REQUIRE(initial_lcao.getBasisSetSize() == 4);
    REQUIRE(initial_lcao.C->rows() == (many ? 4 : 3));
    REQUIRE(initial_lcao.C->cols() == 4);
    const int selected[] = {model == 1 ? 1 : 0, model == 1 ? 0 : 1, 2, 3};
    for (int row = 0; row < (many ? 4 : 3); ++row)
      for (int column = 0; column < 4; ++column)
        CHECK((*initial_lcao.C)(row, column) == (column == selected[row] ? 1.0 : 0.0));
    particles.update();
    ParticleSet ordinary_particles(particles);
    std::unique_ptr<TrialWaveFunction> ordinary, ordinary_sibling;
    if (md)
    {
      // Retain ordinary source/sibling state. Only the detached capture
      // object and its private descendants receive the numerical opt-in.
      ordinary = source->makeClone(ordinary_particles);
      ordinary_sibling = ordinary->makeClone(ordinary_particles);
      isolate(*source);
      source->findMSD().front().get().enableAcceptedReferenceOrbitalVirtualRatios();
      CHECK_FALSE(ordinary->findMSD().front().get().usesAcceptedReferenceOrbitalVirtualRatios());
    }
    source->evaluateLog(particles);
    auto options = adapterOptions();
    if (joint) options.maximum_parameter_count = 17;
    LegacyWaveFunctionAdapter adapter("conventional", *source, options);
    REQUIRE(adapter.snapshotParameters().values.size() == (polarized ? 16 : joint ? 17 : md ? 16 : many ? 8 : 6));
    if (joint)
    {
      REQUIRE(adapter.registrationVariables().name(0) == "CI1");
      REQUIRE(adapter.snapshotParameters().values[0] == 0.1);
    }
    std::array<std::array<int, 2>, 2> history_indices{};
    std::array<int, 2> generated_index{};
    if (polarized)
      for (int spin = 0; spin < 2; ++spin)
      {
        const bool three = particles.groupsize(spin) == 3;
        const auto& variables = rotated(*source, spin).activeRotationVariables();
        REQUIRE(variables.size() == (three ? 5 : 6));
        const auto name = [&](int p, int q) {
          std::ostringstream text;
          text << (spin ? "rot_dn" : "rot") << "_orb_rot_" << std::setfill('0') << std::setw(4)
               << p << '_' << std::setw(4) << q << "_r";
          return text.str();
        };
        for (int pair = 0; pair < 2; ++pair)
        {
          const auto key = name(0, (three ? 2 : 1) + pair);
          const int local = variables.getLoc(key);
          REQUIRE(local >= 0);
          history_indices[spin][pair] = variables.where(local);
          REQUIRE(history_indices[spin][pair] >= 0);
        }
        generated_index[spin] = variables.getLoc(three ? name(2, 3) : name(1, 2));
        REQUIRE(generated_index[spin] >= 0);
      }
    ParticleSet unsupported_j3_particles(particles);
    auto unsupported_j3 = source->makeClone(unsupported_j3_particles);
    addPolynomialJ3(*unsupported_j3, unsupported_j3_particles, ions);
    CHECK_THROWS_WITH(adapter.validateNonLocalECPProfile(*unsupported_j3, unsupported_j3_particles),
        Catch::Matchers::ContainsSubstring("fixed single Slater without active orbitals"));
    ECPComponentBuilder builder(many ? "C" : "Be", OHMMS::Controller, node ? 3 : 2, 1);
    const auto asset = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() /
        (many ? "tests/pseudopotentials_for_tests/C.BFD.xml" : "tests/pseudopotentials_for_tests/Be.BFD.xml");
    REQUIRE(builder.read_pp_file(asset.string()));
    auto nonlocal = std::make_unique<NonLocalECPotential>(ions, particles, false, true);
    auto* nl = nonlocal.get();
    nonlocal->addComponent(0, std::move(builder.pp_nonloc));
    auto local = std::make_unique<LocalECPotential>(ions, particles);
    auto* loc = local.get();
    local->add(0, std::move(builder.pp_loc), builder.Zeff);
    QMCHamiltonian ham;
    ham.addOperator(std::make_unique<BareKineticEnergy>(particles), "Kinetic");
    ham.addOperator(std::make_unique<CoulombPotential>(particles, true, false), "ElecElec");
    ham.addOperator(std::move(local), "LocalECP");
    ham.addOperator(std::move(nonlocal), "NonLocalECP");
    ham.addObservables(particles);
    StdRandom<QMCTraits::FullPrecRealType> rng(1931);
    class CountingZeroRandom final : public FakeRandom<QMCTraits::FullPrecRealType>
    {
    public:
      QMCTraits::FullPrecRealType operator()() override { ++draws; return 0; }
      std::size_t draws = 0;
    } zero_rng;
    nl->setRandomGenerator(!node ? static_cast<RandomBase<QMCTraits::FullPrecRealType>*>(&rng) : &zero_rng);
    ResourceCollection ps_resources("orb_ecp_P"), wf_resources("orb_ecp_WF"), ham_resources("orb_ecp_H");
    particles.createResource(ps_resources);
    source->createResource(wf_resources);
    ham.createResource(ham_resources);
    const auto positions = particles.R;
    const auto ion_positions = ions.R;
    for (int step = 0; step < (model == 0 || model == 3 || model == 5 || polarized || csf ? 3 : 1); ++step)
    {
      CAPTURE(step);
      if (step > 0)
      {
        auto candidate = adapter.snapshotParameters();
        const int offset = joint ? 1 : 0;
        if (joint) candidate.values[0] += 0.002;
        if (polarized)
          for (int spin = 0; spin < 2; ++spin)
          {
            candidate.values[history_indices[spin][0]] += step == 1 ? .03 : -.02;
            candidate.values[history_indices[spin][1]] += step == 1 ? -.01 : .04;
          }
        else
        {
          candidate.values[offset] += step == 1 ? 0.03 : -0.02;
          candidate.values[offset + 1] += step == 1 ? -0.01 : 0.04;
          candidate.values[offset + 2] += 0.002;
          if (md)
          {
            candidate.values[offset + 6] += step == 1 ? -0.025 : 0.015;
            candidate.values[offset + 7] += step == 1 ? 0.01 : -0.035;
          }
        }
        adapter.installSnapshot(*source, particles, candidate);
        adapter.publishParameters(candidate, candidate.version);
        if (step == 2 && !md)
          CHECK(rotated(*source).fullRotationCoordinates()[2] != 0.0);
        if (step == 2 && md)
          for (int spin = 0; spin < 2; ++spin)
            CHECK(rotated(*source, spin).fullRotationCoordinates()[polarized ? generated_index[spin] : 3] != 0.0);
      }
      const auto snapshot = std::make_shared<const StructuredParameterSnapshot>(adapter.snapshotParameters());
      const auto source_log = source->getLogPsi();
      const auto source_energy = kineticEnergy(particles);
      auto& source_lcao = dynamic_cast<LCAOrbitalSet&>(*rotated(*source).Phi_);
      const auto source_coefficients = *source_lcao.C;
      const auto source_reference = source_lcao.originalRotationCoefficients();
      const auto source_full = rotated(*source).fullRotationCoordinates();
      const auto down_coefficients = md ? *dynamic_cast<LCAOrbitalSet&>(*rotated(*source, 1).Phi_).C : source_coefficients;
      const auto down_reference = md ? dynamic_cast<LCAOrbitalSet&>(*rotated(*source, 1).Phi_).originalRotationCoefficients() : source_reference;
      const auto down_full = md ? rotated(*source, 1).fullRotationCoordinates() : source_full;
      const auto source_ci = md ? source->findMSD().front().get().get_C() : std::vector<QMCTraits::ValueType>{};
      const auto source_map = md ? source->findMSD().front().get().get_C2node() : std::vector<std::vector<std::size_t>>{};
      if (csf)
      {
        REQUIRE(source->findMSD().front().get().hasCSFExpansion());
        CHECK(source->findMSD().front().get().fixedCoefficientReference() == QMCTraits::ValueType{.6});
        REQUIRE(source_ci.size() == 3);
        CHECK(source_ci[0] == QMCTraits::ValueType{.48});
        CHECK((source_map == std::vector<std::vector<std::size_t>>{{0, 0, 1}, {0, 1, 0}}));
        checkClose(derivativeValue(source_ci[1]), snapshot->values[0] * std::sqrt(.5));
        checkClose(derivativeValue(source_ci[2]), -snapshot->values[0] * std::sqrt(.5));
      }
      if (joint)
      {
        // One candidate jointly changes CI and BOTH frames on detached banks.
        ParticleSet candidate_particles(particles);
        auto candidate = source->makeClone(candidate_particles);
        isolate(*candidate);
        auto point = *snapshot;
        point.values[0] += .003;
        point.values[polarized ? history_indices[0][0] : 1] += .01;
        point.values[polarized ? history_indices[1][0] : 7] -= .02;
        candidate_particles.update(); adapter.installSnapshot(*candidate, candidate_particles, point);
        const auto& ci = candidate->findMSD().front().get().get_C();
        CHECK(ci[0] == QMCTraits::ValueType(csf ? .48 : 1));
        if (csf)
        {
          REQUIRE(ci.size() == 3);
          checkClose(derivativeValue(ci[1]), point.values[0] * std::sqrt(.5));
          checkClose(derivativeValue(ci[2]), -point.values[0] * std::sqrt(.5));
          CHECK(candidate->findMSD().front().get().fixedCoefficientReference() == QMCTraits::ValueType{.6});
          CHECK(candidate->findMSD().front().get().get_C2node() == source_map);
        }
        else
          CHECK(std::real(ci[1]) == point.values[0]);
        CHECK(candidate->findMSD().front().get().get_C().data() != source->findMSD().front().get().get_C().data());
        CHECK(rotated(*candidate).fullRotationCoordinates() != source_full);
        CHECK(rotated(*candidate, 1).fullRotationCoordinates() != down_full);
        CHECK(source->findMSD().front().get().get_C() == source_ci);
        CHECK(*source_lcao.C == source_coefficients);
        CHECK(*dynamic_cast<LCAOrbitalSet&>(*rotated(*source, 1).Phi_).C == down_coefficients);
      }
      ParticleSet private_particles(particles), sibling_particles(particles);
      auto evaluator = source->makeClone(private_particles), sibling = source->makeClone(sibling_particles);
      private_particles.update();
      sibling_particles.update();
      isolate(*evaluator);
      sibling->evaluateLog(sibling_particles);
      const auto sibling_log = sibling->getLogPsi();
      const auto sibling_energy = kineticEnergy(sibling_particles);
      class RecoverableReplay final : public LegacyWaveFunctionReplay
      {
      public:
        RecoverableReplay(TrialWaveFunction& wf, ParticleSet& p) : wf(wf), p(p), positions(p.R) {}
        std::size_t sampleCount() const noexcept override { return 1; }
        LegacyWaveFunctionReplaySample prepareSample(std::size_t index) override
        {
          if (until_failure == 0) { until_failure = -1; throw std::runtime_error("private replay failure"); }
          if (until_failure > 0) --until_failure;
          REQUIRE(index == 0);
          p.R = positions;
          p.update();
          return {wf, p};
        }
        int until_failure = -1;
        TrialWaveFunction& wf;
        ParticleSet& p;
        ParticleSet::ParticlePos positions;
      };
      auto replay = std::make_shared<RecoverableReplay>(*evaluator, private_particles);
      auto op = adapter.makeDerivativeOperator(replay, snapshot, 2, 5);
      auto consumer = adapter.makeNonLocalECPDerivativeConsumer(replay, snapshot, 2, 5);
      const auto ecp_mask = localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);
      const auto total_mask = ecp_mask | localEnergyTermBit(LocalEnergyTerm::KINETIC);
      const std::vector<DerivativeValue> unit{1};
      const VJPCoefficientChannel capture{"capture", DerivativeProduct::LOCAL_ENERGY_VJP,
          coefficientsFor(*op, unit), ecp_mask};
      RecordingParameterSink idle;
      RefVectorWithLeader<ParticleSet> ps(particles, {particles});
      RefVectorWithLeader<TrialWaveFunction> wfs(*source, {*source});
      RefVectorWithLeader<QMCHamiltonian> hs(ham, {ham});
      QMCHamiltonian::StreamingNonLocalECPResult accepted;
      {
        ResourceCollectionTeamLock<ParticleSet> p_lock(ps_resources, ps);
        ResourceCollectionTeamLock<TrialWaveFunction> w_lock(wf_resources, wfs);
        ResourceCollectionTeamLock<QMCHamiltonian> h_lock(ham_resources, hs);
        accepted = QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(hs, wfs, ps, *consumer, {&capture, 1}, idle);
      }
      const auto rng_after_capture = rng;
      const auto zero_draws_after_capture = zero_rng.draws;
      if (node) CHECK(zero_draws_after_capture > 0);
      const auto quadrature = nl->captureQuadratureState();
      auto record = consumer->releaseJVPReplay();
      REQUIRE(record->gridFingerprint() == accepted.grid_fingerprint);
      const std::vector<NonLocalECPReplayPoint> points(record->points().begin(), record->points().end());
      REQUIRE(points.size() >= 2);
      std::size_t md_node = points.size();
      if (node)
      {
        std::ostringstream diagnostic;
        diagnostic << std::setprecision(17);
        for (const auto& point : points)
        {
          private_particles.R = positions;
          for (int d = 0; d < 3; ++d) private_particles.R[point.electron][d] = point.replacement_position[d];
          private_particles.update();
          SPOSet::ValueVector values(many ? 4 : 3);
          rotated(*evaluator).Phi_->evaluateValue(private_particles, point.electron, values);
          diagnostic << "electron=" << point.electron << " q=" << point.replacement_position[0] << ','
                     << point.replacement_position[1] << ',' << point.replacement_position[2]
                     << " captured_ratio=" << point.complete_ratio << " occupied=" << values[0]
                     << " virtuals=" << values[1] << ',' << values[2] << '\n';
          if (md && md_node == points.size())
          {
            SPOSet::ValueMatrix occupied(2, 4), laps(2, 4);
            SPOSet::GradMatrix gradients(2, 4);
            const int spin = point.electron / 2;
            rotated(*evaluator, spin).evaluate_notranspose(private_particles, 2 * spin, 2 * spin + 2,
                                                          occupied, gradients, laps);
            if (occupied(0, 0) * occupied(1, 1) - occupied(0, 1) * occupied(1, 0) == QMCTraits::ValueType{} &&
                occupied(0, 0) * occupied(1, 2) - occupied(0, 2) * occupied(1, 0) == QMCTraits::ValueType{})
              md_node = &point - points.data();
          }
        }
        INFO(diagnostic.str());
        if (md)
        {
          // Exact independent node; the bordered ratio may retain roundoff.
          REQUIRE(md_node < points.size());
          checkClose(points[md_node].complete_ratio, {});
        }
        else
          REQUIRE(std::any_of(points.begin(), points.end(), [](const auto& point) { return point.complete_ratio == DerivativeValue{}; }));
        replay->prepareSample(0);
      }
      op->bindNonLocalECPJVPReplay(std::move(record));
      CHECK(op->storageDiagnostics().real_parameter_vectors == 2);
      double fixed_potential = loc->evaluate(particles);
      for (int electron = 0; electron < particles.getTotalNum(); ++electron)
        for (int other = 0; other < electron; ++other)
          fixed_potential += 1.0 / std::sqrt(dot(positions[electron] - positions[other], positions[electron] - positions[other]));
      // Independent physical MD numerator: signed occupied 2x2/3x3 matrices,
      // fixed CI contraction and separately evaluated J, including at nodes.
      const auto dense_value = [&](TrialWaveFunction& wf, ParticleSet& p, double actual_csf1) {
        std::array<QMCTraits::ValueType, 2> products{1, 1};
        std::array<std::array<QMCTraits::ValueType, 2>, 2> occupied{};
        for (int spin = 0; spin < 2; ++spin)
        {
          const int n = p.groupsize(spin), first = p.first(spin);
          SPOSet::ValueMatrix values(n, 4), laps(n, 4);
          SPOSet::GradMatrix gradients(n, 4);
          rotated(wf, spin).evaluate_notranspose(p, first, first + n, values, gradients, laps);
          for (int term = 0; term < 2; ++term)
          {
            const int last = n - 1 + term;
            const auto determinant = n == 2 ? values(0, 0) * values(1, last) - values(0, last) * values(1, 0) :
                values(0, 0) * (values(1, 1) * values(2, last) - values(1, last) * values(2, 1)) -
                values(0, 1) * (values(1, 0) * values(2, last) - values(1, last) * values(2, 0)) +
                values(0, last) * (values(1, 0) * values(2, 1) - values(1, 1) * values(2, 0));
            products[term] *= determinant;
            occupied[spin][term] = determinant;
          }
        }
        double j = 0;
        p.G = QMCTraits::GradType{}; p.L = QMCTraits::ValueType{};
        for (auto& component : wf.getOrbitals())
          if (!component->isFermionic()) j += component->evaluateLog(p, p.G, p.L).real();
        if (csf)
        {
          // Independent literal XML weights with ACTUAL CSF variables, not expanded get_C.
          const auto actual_csf0 = wf.findMSD().front().get().fixedCoefficientReference();
          return (actual_csf0 * .8 * occupied[0][0] * occupied[1][0] +
              actual_csf1 * std::sqrt(.5) * (occupied[0][0] * occupied[1][1] -
                                            occupied[0][1] * occupied[1][0])) * std::exp(j);
        }
        const auto& c = wf.findMSD().front().get().get_C();
        return (c[0] * products[0] + c[1] * products[1]) * std::exp(j);
      };
      if (csf)
      {
        // Independent spatial finite difference of complete signed Psi checks kinetic energy.
        ParticleSet spatial_particles(particles);
        auto spatial = source->makeClone(spatial_particles);
        isolate(*spatial);
        spatial_particles.update(); adapter.installSnapshot(*spatial, spatial_particles, *snapshot);
        const auto psi = dense_value(*spatial, spatial_particles, snapshot->values[0]);
        QMCTraits::ValueType laplacian{};
        constexpr double h = 1e-4;
        for (int electron = 0; electron < 4; ++electron)
          for (int dimension = 0; dimension < 3; ++dimension)
          {
            spatial_particles.R = positions; spatial_particles.R[electron][dimension] += h; spatial_particles.update();
            const auto plus = dense_value(*spatial, spatial_particles, snapshot->values[0]);
            spatial_particles.R = positions; spatial_particles.R[electron][dimension] -= h; spatial_particles.update();
            laplacian += plus + dense_value(*spatial, spatial_particles, snapshot->values[0]) - 2.0 * psi;
          }
        CHECK(std::real(-.5 * laplacian / (h * h * psi)) == Catch::Approx(source_energy.real()).epsilon(3e-5).margin(3e-6));
      }
      // Independent signed complete-ratio FD oracle uses ordinary proposed
      // moves and rejects them, never a virtual-point inverse/logarithm.
      const auto fixed_grid_energy = [&](const StructuredParameterSnapshot& candidate, bool kinetic) {
        ParticleSet p(particles);
        auto wf = source->makeClone(p);
        p.update();
        isolate(*wf);
        adapter.installSnapshot(*wf, p, candidate);
        DerivativeValue energy = kinetic ? kineticEnergy(p) + fixed_potential : DerivativeValue{};
        if (md)
        {
          const auto accepted_value = dense_value(*wf, p, csf ? candidate.values[0] : 0);
          for (const auto& point : points)
          {
            p.R = positions;
            for (int d = 0; d < 3; ++d) p.R[point.electron][d] = point.replacement_position[d];
            p.update();
            energy += point.bare_weight * derivativeValue(dense_value(*wf, p, csf ? candidate.values[0] : 0) / accepted_value);
          }
          return energy;
        }
        for (const auto& point : points)
        {
          auto replacement = p.R[point.electron];
          for (int d = 0; d < 3; ++d) replacement[d] = point.replacement_position[d];
          p.makeMove(point.electron, replacement - p.R[point.electron]);
          energy += point.bare_weight * derivativeValue(wf->calcRatio(p, point.electron));
          wf->rejectMove(point.electron);
          p.rejectMove(point.electron);
        }
        return energy;
      };
      CHECK(fixed_grid_energy(*snapshot, true).real() == Catch::Approx(accepted.local_energies[0]).epsilon(3e-12).margin(3e-12));
      const auto reference_log = [&](const StructuredParameterSnapshot& candidate) {
        ParticleSet p(particles);
        auto wf = source->makeClone(p);
        p.update();
        isolate(*wf);
        adapter.installSnapshot(*wf, p, candidate);
        return md ? std::log(std::abs(dense_value(*wf, p, csf ? candidate.values[0] : 0))) : wf->getLogPsi();
      };
      const std::array<VJPCoefficientChannel, 3> channels{{
          {"total", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(*op, unit), total_mask},
          {"ecp", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(*op, unit), ecp_mask},
          {"score", DerivativeProduct::SCORE_VJP, coefficientsFor(*op, unit), 0}}};
      RecordingParameterSink rows, hermitian;
      op->applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::TRANSPOSE, rows);
      op->applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::HERMITIAN, hermitian);
      std::vector<DerivativeValue> direction(snapshot->values.size());
      DerivativeValue dual{}, score_dual{};
      for (std::size_t parameter = 0; parameter < snapshot->values.size(); ++parameter)
      {
        CAPTURE(parameter);
        auto plus = *snapshot, minus = *snapshot;
        plus.values[parameter] += 1e-5;
        minus.values[parameter] -= 1e-5;
        for (int channel = 0; channel < 2; ++channel)
        {
          const auto fd = (fixed_grid_energy(plus, channel == 0) - fixed_grid_energy(minus, channel == 0)) / 2e-5;
          CHECK(rows.result()[channel][parameter].real() == Catch::Approx(fd.real()).epsilon(3e-6).margin(3e-8));
          checkClose(hermitian.result()[channel][parameter], std::conj(rows.result()[channel][parameter]));
        }
        CHECK(rows.result()[2][parameter].real() ==
              Catch::Approx((reference_log(plus) - reference_log(minus)) / 2e-5).epsilon(3e-6).margin(3e-8));
        direction[parameter] = 0.1 * (parameter + 1);
        dual += direction[parameter] * rows.result()[0][parameter];
        score_dual += direction[parameter] * rows.result()[2][parameter];
      }
      if (node && md)
      {
        // A zero accepted spatial-reference derivative must use only the
        // private fallback. Independently difference the full physical Psi,
        // not log(Psi), to check its complete kinetic energy.
        bool zero_spatial_reference = false;
        for (int spin = 0; spin < 2; ++spin)
        {
          const auto& det = source->findMSD().front().get().getDet(spin);
          CHECK(det.usesPrivateAcceptedReferenceOrbitalRows());
          CHECK_FALSE(ordinary->findMSD().front().get().getDet(spin).usesPrivateAcceptedReferenceOrbitalRows());
          CHECK_FALSE(ordinary_sibling->findMSD().front().get().getDet(spin).usesPrivateAcceptedReferenceOrbitalRows());
          for (int i = 0; i < 2; ++i)
            for (int d = 0; d < 3; ++d) zero_spatial_reference |= det.getGrads()(0, i)[d] == QMCTraits::ValueType{};
        }
        REQUIRE(zero_spatial_reference);
        ParticleSet spatial_particles(particles);
        auto spatial = source->makeClone(spatial_particles);
        isolate(*spatial);
        spatial_particles.update(); adapter.installSnapshot(*spatial, spatial_particles, *snapshot);
        const auto psi = dense_value(*spatial, spatial_particles, 0);
        QMCTraits::ValueType laplacian{};
        constexpr double h = 1e-4;
        for (int electron = 0; electron < 4; ++electron)
          for (int dimension = 0; dimension < 3; ++dimension)
          {
            spatial_particles.R = positions; spatial_particles.R[electron][dimension] += h; spatial_particles.update();
            const auto plus = dense_value(*spatial, spatial_particles, 0);
            spatial_particles.R = positions; spatial_particles.R[electron][dimension] -= h; spatial_particles.update();
            laplacian += plus + dense_value(*spatial, spatial_particles, 0) - 2.0 * psi;
          }
        CHECK(std::real(-0.5 * laplacian / (h * h * psi)) == Catch::Approx(source_energy.real()).epsilon(3e-5).margin(3e-6));
        REQUIRE(md_node < points.size());
        const auto zero = points.begin() + md_node;
        replay->prepareSample(0);
        adapter.installSnapshot(*evaluator, private_particles, *snapshot);
        for (int d = 0; d < 3; ++d) private_particles.R[zero->electron][d] = zero->replacement_position[d];
        private_particles.update();
        SPOSet::ValueMatrix values(2, 4), laps(2, 4);
        SPOSet::GradMatrix gradients(2, 4);
        const int spin = zero->electron / 2;
        rotated(*evaluator, spin).evaluate_notranspose(private_particles, 2 * spin, 2 * spin + 2, values, gradients, laps);
        for (int term = 0; term < 2; ++term)
          CHECK(values(0, 0) * values(1, 1 + term) - values(0, 1 + term) * values(1, 0) == QMCTraits::ValueType{});
        std::vector<QMCTraits::ValueType> derivative(snapshot->values.size());
        checkClose(derivativeValue(evaluator->findMSD().front().get().evaluateAcceptedReferenceOrbitalRatioDerivatives(
            private_particles, zero->electron, zero->electron, adapter.registrationVariables(),
            {derivative.data(), derivative.size()})), {});
        for (const auto value : derivative) CHECK((std::isfinite(std::real(value)) && std::isfinite(std::imag(value))));
        CHECK(std::any_of(derivative.begin(), derivative.end(), [](auto x) { return std::abs(x) > 1e-8; }));
        // Accepted reference is regular but its py excitation is exactly zero.
        replay->prepareSample(0);
        private_particles.R[0] = {1, 0, 0}; private_particles.update();
        evaluator->evaluateLog(private_particles);
        CHECK_THROWS_WITH(evaluator->findMSD().front().get().validateAcceptedOrbitalDerivativeState(),
            "Private orbital ECP accepted excitation matrix is singular or nonfinite");
        replay->prepareSample(0);
      }
      else if (model == 1)
      {
        // The zero-ratio knot itself contributes a finite NONZERO derivative.
        const auto zero = std::find_if(points.begin(), points.end(), [](const auto& p) { return p.complete_ratio == DerivativeValue{}; });
        replay->prepareSample(0);
        adapter.installSnapshot(*evaluator, private_particles, *snapshot);
        SPOSet::ValueVector reference(3), replacement(3), derivative(snapshot->values.size());
        rotated(*evaluator).Phi_->evaluateValue(private_particles, zero->electron, reference);
        for (int d = 0; d < 3; ++d) private_particles.R[zero->electron][d] = zero->replacement_position[d];
        private_particles.update();
        rotated(*evaluator).Phi_->evaluateValue(private_particles, zero->electron, replacement);
        derivative = 0;
        CHECK(rotated(*evaluator).evaluateOneOccupiedRatioDerivatives(reference, replacement, adapter.registrationVariables(), derivative) == QMCTraits::ValueType{});
        CHECK(std::abs(derivative[0]) > 1e-8);
        reference[0] = std::numeric_limits<double>::quiet_NaN();
        CHECK_THROWS(rotated(*evaluator).evaluateOneOccupiedRatioDerivatives(reference, replacement, adapter.registrationVariables(), derivative));
        reference = 0;
        replacement = 0;
        derivative = 0;
        reference[0] = 0.5;
        replacement[1] = std::numeric_limits<double>::max() / 2;
        derivative[0] = std::numeric_limits<double>::max();
        CHECK_THROWS(rotated(*evaluator).evaluateOneOccupiedRatioDerivatives(reference, replacement, adapter.registrationVariables(), derivative));
      }
      StructuredParameterVectorConstView vector(adapter.parameterSchema(), snapshot->version, {direction.data(), direction.size()});
      RecordingSampleSink first, second, score;
      op->applyLocalEnergyJVP(vector, total_mask, first);
      op->applyLocalEnergyJVP(vector, total_mask, second);
      op->applyScoreJVP(vector, score);
      checkClose(first.result()[0], dual);
      checkClose(score.result()[0], score_dual);
      CHECK(first.result() == second.result());
      replay->until_failure = 2;
      RecordingParameterSink failed, recovered;
      CHECK_THROWS_WITH(op->applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::TRANSPOSE, failed),
                        Catch::Matchers::ContainsSubstring("private replay failure"));
      CHECK(failed.state() == DerivativeSinkState::POISONED);
      op->applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::TRANSPOSE, recovered);
      CHECK(recovered.result() == rows.result());
      auto expected_rng = rng_after_capture;
      CHECK(rng() == expected_rng());
      CHECK(zero_rng.draws == zero_draws_after_capture);
      CHECK(nl->captureQuadratureState().coordinates == quadrature.coordinates);
      source->evaluateLog(particles);
      sibling->evaluateLog(sibling_particles);
      CHECK(source->getLogPsi() == source_log);
      CHECK(kineticEnergy(particles) == source_energy);
      CHECK(sibling->getLogPsi() == sibling_log);
      CHECK(kineticEnergy(sibling_particles) == sibling_energy);
      CHECK(*source_lcao.C == source_coefficients);
      CHECK(source_lcao.originalRotationCoefficients() == source_reference);
      CHECK(rotated(*source).fullRotationCoordinates() == source_full);
      if (md)
      {
        CHECK(source->findMSD().front().get().get_C() == source_ci);
        CHECK(source_ci[0] == QMCTraits::ValueType(csf ? .48 : 1));
        CHECK(source->findMSD().front().get().get_C2node() == source_map);
        CHECK(sibling->findMSD().front().get().get_C() == source_ci);
        if (csf)
        {
          const std::vector<QMCTraits::ValueType> original_ci{.48, .1 * std::sqrt(.5), -.1 * std::sqrt(.5)};
          CHECK(ordinary->findMSD().front().get().get_C() == original_ci);
          CHECK(source->findMSD().front().get().fixedCoefficientReference() == QMCTraits::ValueType{.6});
          CHECK(sibling->findMSD().front().get().fixedCoefficientReference() == QMCTraits::ValueType{.6});
          CHECK(ordinary->findMSD().front().get().get_C2node() == source_map);
          // With no selected updates, rebuild expanded C from its ACTUAL private CSF bank.
          // This exposes accidental CSF-bank sharing even if the expanded vector stayed detached.
          auto& source_msd = source->findMSD().front().get();
          OptVariables inactive;
          source_msd.checkOutVariables(inactive);
          source_msd.resetParametersExclusive(inactive);
          source_msd.checkOutVariables(adapter.registrationVariables());
          CHECK(source_msd.get_C() == source_ci);
        }
        else
          CHECK((ordinary->findMSD().front().get().get_C() == std::vector<QMCTraits::ValueType>{1, .1}));
        CHECK(ordinary_sibling->findMSD().front().get().get_C() == ordinary->findMSD().front().get().get_C());
        auto& down = dynamic_cast<LCAOrbitalSet&>(*rotated(*source, 1).Phi_);
        CHECK(*down.C == down_coefficients);
        CHECK(down.originalRotationCoefficients() == down_reference);
        CHECK(rotated(*source, 1).fullRotationCoordinates() == down_full);
        CHECK_FALSE(ordinary->findMSD().front().get().usesAcceptedReferenceOrbitalVirtualRatios());
        CHECK_FALSE(ordinary_sibling->findMSD().front().get().usesAcceptedReferenceOrbitalVirtualRatios());
        for (int spin = 0; spin < 2; ++spin)
        {
          CHECK(dynamic_cast<LCAOrbitalSet&>(*rotated(*ordinary, spin).Phi_).C !=
                dynamic_cast<LCAOrbitalSet&>(*rotated(*source, spin).Phi_).C);
          const auto& original = dynamic_cast<LCAOrbitalSet&>(*rotated(*ordinary, spin).Phi_);
          for (int row = 0; row < 4; ++row)
            for (int column = 0; column < 4; ++column)
            {
              CHECK((*original.C)(row, column) == (row == column ? 1.0 : 0.0));
              CHECK(original.originalRotationCoefficients()(row, column) == (row == column ? 1.0 : 0.0));
            }
          CHECK(original.C == dynamic_cast<LCAOrbitalSet&>(*rotated(*ordinary_sibling, spin).Phi_).C);
          CHECK(rotated(*ordinary, spin).fullRotationCoordinates() == std::vector<double>(6, 0));
        }
      }
      for (int e = 0; e < particles.getTotalNum(); ++e)
        for (int d = 0; d < 3; ++d) CHECK(particles.R[e][d] == positions[e][d]);
      for (int ion = 0; ion < ions.getTotalNum(); ++ion) CHECK(ions.R[ion] == ion_positions[ion]);
    }
  }
}

TEST_CASE("external LCAO file-image reader and bounded products", "[legacy_adapter][external_lcao]")
{
  const int model = GENERATE(0, 1, 2, 3, 4);
  const bool ecp = model != 0, active = model >= 2;
  CAPTURE(model);
  ParticleSetPool pool(OHMMS::Controller);
  auto ion_owner = std::make_unique<ParticleSet>(pool.getSimulationCell());
  auto electron_owner = std::make_unique<ParticleSet>(pool.getSimulationCell());
  auto& ions = *ion_owner;
  auto& particles = *electron_owner;
  ions.setName("ion0");
  ions.create({1, 1});
  ions.R[0] = {0, 0, 0};
  ions.R[1] = {0, 0, 3.0139239778};
  auto& ion_species = ions.getSpeciesSet();
  const int li = ion_species.addSpecies("Li"), h = ion_species.addSpecies("H");
  const int charge = ion_species.addAttribute("charge");
  ion_species(charge, li) = ecp ? 1 : 3;
  ion_species(charge, h) = 1;
  if (ecp)
  {
    const int atomic_number = ion_species.addAttribute("atomicnumber");
    ion_species(atomic_number, li) = 3; ion_species(atomic_number, h) = 1;
  }
  ions.resetGroups();
  particles.setName("e");
  particles.create(ecp ? std::vector<int>{1, 1} : std::vector<int>{2, 2});
  particles.R[0] = {0.2, -0.3, 0.4};
  particles.R[1] = {-0.5, 0.1, ecp ? 0.7 : 1.7};
  if (!ecp)
  {
    particles.R[2] = {0.4, 0.2, -0.7};
    particles.R[3] = {-0.2, -0.4, 2.9};
  }
  auto& species = particles.getSpeciesSet();
  const int u = species.addSpecies("u"), d = species.addSpecies("d");
  const int mass = species.addAttribute("mass"), electron_charge = species.addAttribute("charge");
  species(mass, u) = species(mass, d) = 1;
  species(electron_charge, u) = species(electron_charge, d) = -1;
  particles.resetGroups();
  pool.addParticleSet(std::move(ion_owner));
  pool.addParticleSet(std::move(electron_owner));
  const auto asset = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() /
      (ecp ? "tests/molecules/LiH_pp/LiH.orbs.h5" : "tests/molecules/LiH_ae_pyscf/LiH-pyscf.h5");
  const auto temporary = std::filesystem::temp_directory_path() /
      ("lcao_identity_" + std::to_string(OHMMS::Controller->rank()) + "_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".h5");
  FileCleanup cleanup(temporary);
  if (!active) std::filesystem::copy_file(asset, temporary);
  const auto xml = [&](const std::filesystem::path& file, std::size_t cap = 1048576) {
    std::string input = "<wavefunction name='psi0' target='e'><sposet_collection type='MolecularOrbital' name='LCAOBSet' "
        "source='ion0' transform='yes' source_identity='file_image' source_identity_max_bytes='" +
        std::to_string(cap) + "' href='" + file.string() + R"('>
        <sposet name="spo-ud" size="2"><occupation mode="ground"/><coefficient size="135" spindataset="0"/></sposet>
      </sposet_collection><determinantset><slaterdeterminant>
        <determinant sposet="spo-ud"/><determinant sposet="spo-ud"/>
      </slaterdeterminant></determinantset>
      <jastrow name="J2" type="Two-Body" function="Bspline">
        <correlation rcut="4" size="4" speciesA="u" speciesB="u">
          <coefficients id="uu" type="Array" optimize="yes">0.04 -0.02 0.01 -0.005</coefficients>
        </correlation><correlation rcut="4" size="4" speciesA="u" speciesB="d">
          <coefficients id="ud" type="Array" optimize="yes">0.12 -0.08 0.04 -0.01</coefficients>
        </correlation>
      </jastrow></wavefunction>)";
    if (ecp)
    {
      for (const auto& [before, after] : std::array<std::pair<std::string, std::string>, 2>{{
          {"size=\"2\"", "size=\"1\""}, {"size=\"135\"", "size=\"62\""}}})
        input.replace(input.find(before), before.size(), after);
      const auto begin = input.find("<correlation"), end = input.find("</correlation>", begin);
      input.erase(begin, end + std::string("</correlation>").size() - begin);
    }
    if (active)
    {
      const auto begin = input.find("<sposet name="), end = input.find("</sposet>", begin) + 9;
      input.replace(begin, end - begin, R"(<rotated_sposet name="spo-ud"><sposet name="base" size="3">
        <occupation mode="ground"/><coefficient size="62" spindataset="0"/>
        </sposet></rotated_sposet>)");
    }
    return input;
  };
  const auto build = [&](Libxml2Document& document, const std::string& input) {
    REQUIRE(document.parseFromString(input));
    WaveFunctionFactory factory(particles, pool.getPool(), OHMMS::Controller);
    auto result = factory.buildTWF(document.getRoot(), RuntimeOptions{});
    REQUIRE(result);
    result->storeXMLNode(document.getRoot());
    return result;
  };
  Libxml2Document definition, relocated_definition;
  auto source = build(definition, xml(asset));
  auto relocated = build(relocated_definition, xml(active ? asset : temporary));
  const auto rotation = [](TrialWaveFunction& wf) -> RotatedSPOs& {
    auto& sd = wf.findSD()[0].get();
    REQUIRE(&sd.getPhi(0) == &sd.getPhi(1));
    return dynamic_cast<RotatedSPOs&>(sd.getPhi(0));
  };
  const auto isolate = [&](TrialWaveFunction& wf) {
    if (active) rotation(wf).isolateOptimizationState();
  };
  const auto leaf = [&](TrialWaveFunction& wf) -> LCAOrbitalSet& {
    return dynamic_cast<LCAOrbitalSet&>(active ? *rotation(wf).Phi_ : wf.findSD()[0].get().getPhi(0));
  };
  isolate(*source);
  const auto& lcao = leaf(*source);
  REQUIRE(lcao.fileImageIdentity());
  CHECK(lcao.fileImageIdentity()->byteCount() == std::filesystem::file_size(asset));
  // LCAO distance tables read the source's coordinate storage, not just R.
  // Initialize it explicitly instead of depending on reused heap contents.
  ions.update();
  particles.update();
  source->evaluateLog(particles);
  QMCHamiltonian ham;
  ham.addOperator(std::make_unique<BareKineticEnergy>(particles), "Kinetic");
  if (ecp)
  {
    ham.addOperator(std::make_unique<CoulombPotential>(particles, true, false), "ElecElec");
    ham.addOperator(std::make_unique<CoulombPotential>(ions, false, false), "IonIon");
    Libxml2Document pp;
    const auto directory = asset.parent_path();
    REQUIRE(pp.parseFromString("<pairpot format='xml' algorithm='batched' DLA='no' pbc='no'>"
        "<pseudo elementType='Li' nrule='2' l-local='1' href='" + (directory / "Li.BFD.xml").string() + "'/>"
        "<pseudo elementType='H' nrule='2' l-local='0' href='" + (directory / "H.BFD.xml").string() + "'/></pairpot>"));
    ECPotentialBuilder builder(ham, ions, particles, OHMMS::Controller);
    REQUIRE(builder.put(pp.getRoot()));
    REQUIRE(ham.ecpModelProvenance()->ordinary_locality_vp);
    ham.addObservables(particles);
    particles.update(); source->evaluateLog(particles);
  }
  if (active)
  {
    REQUIRE(lcao.C->rows() == 3);
    REQUIRE(lcao.C->cols() == 72);
    LegacyWaveFunctionAdapter history("legacy/external/history", *source, adapterOptions());
    REQUIRE(history.snapshotParameters().values.size() == 6);
    // Fresh fixture per stage: initial, one rotation, then a noncommuting rotation.
    for (int step = 0; step < model - 2; ++step)
    {
      auto point = history.snapshotParameters();
      point.values[0] += step == 0 ? .03 : -.02;
      point.values[1] += step == 0 ? -.01 : .04;
      history.installSnapshot(*source, particles, point);
      history.publishParameters(point, point.version);
    }
    if (model == 4) CHECK(rotation(*source).fullRotationCoordinates()[2] != 0);
  }
  const double original_log = source->getLogPsi();
  const auto original_energy = kineticEnergy(particles);
  const auto accepted_coefficients = *lcao.C;
  const auto accepted_reference = active ? lcao.originalRotationCoefficients() : LCAOrbitalSet::OffloadValueMatrix{};
  const auto accepted_frame = active ? rotation(*source).fullRotationCoordinates() : std::vector<double>{};
  ParticleSet private_particles(particles), sibling_particles(particles);
  auto evaluator = source->makeClone(private_particles), sibling = source->makeClone(sibling_particles);
  isolate(*evaluator); isolate(*sibling);
  private_particles.update(); sibling_particles.update(); sibling->evaluateLog(sibling_particles);
  const auto sibling_log = sibling->getLogPsi();
  const auto& cloned_lcao = leaf(*evaluator);
  CHECK(cloned_lcao.fileImageIdentity() == lcao.fileImageIdentity());
  if (active)
  {
    CHECK(cloned_lcao.C.get() != lcao.C.get());
    CHECK(leaf(*sibling).C.get() != lcao.C.get());
  }
  TrainingWaveFunctionBinding binding(*source, particles, adapterOptions(), *OHMMS::Controller);
  TrainingWaveFunctionBinding relocated_binding(*relocated, particles, adapterOptions(), *OHMMS::Controller);
  binding.setHamiltonian(ham);
  relocated_binding.setHamiltonian(ham);
  CHECK(binding.modelIdentity() == relocated_binding.modelIdentity());
  const auto exported = temporary.string() + ".parameters.h5";
  FileCleanup export_cleanup(exported);
  if (ecp && !active)
  {
    binding.exportModel(exported);
    CHECK(relocated_binding.readModel(exported).values == binding.snapshotParameters().values);
    auto altered_ham = ham.makeClone(particles, *relocated);
    auto altered_pp = std::make_shared<QMCHamiltonian::ECPModelProvenance>(*ham.ecpModelProvenance());
    altered_pp->scientific_definition += " changed_pp_content";
    altered_ham->setECPModelProvenance(std::move(altered_pp));
    TrainingWaveFunctionBinding changed_pp(*relocated, particles, adapterOptions(), *OHMMS::Controller);
    changed_pp.setHamiltonian(*altered_ham);
    CHECK_THROWS(changed_pp.readModel(exported));
  }
  LegacyWaveFunctionAdapter adapter("legacy/external", *source, adapterOptions());
  const auto snapshot = adapter.snapshotParameters();
  REQUIRE(snapshot.values.size() == (active ? 6 : ecp ? 4 : 8));
  auto replay = std::make_shared<ShiftedLegacyReplay>(*evaluator, private_particles, std::vector<double>{0});
  auto frame = std::make_shared<const StructuredParameterSnapshot>(snapshot);
  auto op = adapter.makeDerivativeOperator(replay, frame, 0, 0);
  std::uint32_t terms = localEnergyTermBit(LocalEnergyTerm::KINETIC);
  std::vector<NonLocalECPReplayPoint> points;
  StdRandom<QMCTraits::FullPrecRealType> rng(1931);
  double fixed_potential = 0, captured_energy = 0;
  if (ecp)
  {
    terms |= localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);
    ham.setRandomGenerator(&rng);
    auto consumer = adapter.makeNonLocalECPDerivativeConsumer(replay, frame, 0, 0);
    const std::vector<DerivativeValue> unit{1};
    const VJPCoefficientChannel capture{"capture", DerivativeProduct::LOCAL_ENERGY_VJP,
        coefficientsFor(*op, unit), localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)};
    ResourceCollection pr("external_ecp_P"), wr("external_ecp_W"), hr("external_ecp_H");
    particles.createResource(pr); source->createResource(wr); ham.createResource(hr);
    RefVectorWithLeader<ParticleSet> ps(particles, {particles});
    RefVectorWithLeader<TrialWaveFunction> ws(*source, {*source});
    RefVectorWithLeader<QMCHamiltonian> hs(ham, {ham});
    RecordingParameterSink idle;
    {
      ResourceCollectionTeamLock<ParticleSet> pl(pr, ps);
      ResourceCollectionTeamLock<TrialWaveFunction> wl(wr, ws);
      ResourceCollectionTeamLock<QMCHamiltonian> hl(hr, hs);
      captured_energy = QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(
          hs, ws, ps, *consumer, {&capture, 1}, idle).local_energies[0];
    }
    auto grid = consumer->releaseJVPReplay();
    points.assign(grid->points().begin(), grid->points().end());
    REQUIRE(!points.empty());
    op->bindNonLocalECPJVPReplay(std::move(grid));
    for (int i = 0; i < ham.size(); ++i)
      if (!dynamic_cast<BareKineticEnergy*>(ham.getComponent(i)) && !dynamic_cast<NonLocalECPotential*>(ham.getComponent(i)))
        fixed_potential += ham.getComponent(i)->evaluate(*source, particles);
  }
  const auto rng_after_capture = rng;
  NonLocalECPotential* nonlocal = nullptr;
  for (int i = 0; i < ham.size(); ++i)
    if (auto* component = dynamic_cast<NonLocalECPotential*>(ham.getComponent(i))) nonlocal = component;
  const auto quadrature = nonlocal ? nonlocal->captureQuadratureState() : NonLocalECPotential::QuadratureState{};
  const auto pp_identity = ecp ? ham.ecpModelProvenance()->scientific_definition : std::string{};
  const auto source_positions = particles.R, ion_positions = ions.R;
  const auto full_energy = [&](const StructuredParameterSnapshot& point) {
    ParticleSet p(particles);
    auto wf = source->makeClone(p);
    isolate(*wf);
    p.update(); adapter.installSnapshot(*wf, p, point);
    DerivativeValue energy = kineticEnergy(p) + fixed_potential;
    for (const auto& knot : points)
    {
      auto replacement = p.R[knot.electron];
      for (int d = 0; d < 3; ++d) replacement[d] = knot.replacement_position[d];
      p.makeMove(knot.electron, replacement - p.R[knot.electron]);
      energy += knot.bare_weight * derivativeValue(wf->calcRatio(p, knot.electron));
      wf->rejectMove(knot.electron); p.rejectMove(knot.electron);
    }
    return energy;
  };
  if (ecp) CHECK(full_energy(snapshot).real() == Catch::Approx(captured_energy).epsilon(3e-12).margin(3e-12));
  const std::vector<DerivativeValue> unit{1};
  std::vector<VJPCoefficientChannel> channels{
      {"score", DerivativeProduct::SCORE_VJP, coefficientsFor(*op, unit), 0},
      {"energy", DerivativeProduct::LOCAL_ENERGY_VJP, coefficientsFor(*op, unit), terms}};
  if (active) channels.push_back({"kinetic", DerivativeProduct::LOCAL_ENERGY_VJP,
      coefficientsFor(*op, unit), localEnergyTermBit(LocalEnergyTerm::KINETIC)});
  RecordingParameterSink rows;
  op->applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::TRANSPOSE, rows);
  RecordingParameterSink hermitian;
  op->applyVJPs({channels.data(), channels.size()}, DerivativeAdjoint::HERMITIAN, hermitian);
  std::vector<DerivativeValue> direction(snapshot.values.size(), 0.125);
  const StructuredParameterVectorConstView vector(adapter.parameterSchema(), op->parameterVersion(),
                                                 {direction.data(), direction.size()});
  RecordingSampleSink score_jvp, energy_jvp, kinetic_jvp;
  op->applyScoreJVP(vector, score_jvp);
  op->applyLocalEnergyJVP(vector, terms, energy_jvp);
  if (active) op->applyLocalEnergyJVP(vector, localEnergyTermBit(LocalEnergyTerm::KINETIC), kinetic_jvp);
  DerivativeValue score_dual{}, energy_dual{}, kinetic_dual{};
  for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
  {
    for (std::size_t channel = 0; channel < channels.size(); ++channel)
      checkClose(hermitian.result()[channel][parameter], std::conj(rows.result()[channel][parameter]));
    score_dual += direction[parameter] * rows.result()[0][parameter];
    energy_dual += direction[parameter] * rows.result()[1][parameter];
    if (active) kinetic_dual += direction[parameter] * rows.result()[2][parameter];
  }
  checkClose(score_jvp.result()[0], score_dual);
  checkClose(energy_jvp.result()[0], energy_dual);
  if (active) checkClose(kinetic_jvp.result()[0], kinetic_dual);
  for (std::size_t parameter = 0; parameter < snapshot.values.size(); ++parameter)
  {
    auto plus = snapshot, minus = snapshot;
    plus.values[parameter] += 1e-5;
    minus.values[parameter] -= 1e-5;
    // ECP JVP leaves the worker at its last virtual point, not the sample.
    replay->prepareSample(0);
    if (ecp) for (int electron = 0; electron < particles.getTotalNum(); ++electron)
      CHECK(private_particles.R[electron] == source_positions[electron]);
    adapter.installSnapshot(*evaluator, private_particles, plus);
    const double plus_log = evaluator->getLogPsi();
    const auto plus_kinetic = kineticEnergy(private_particles);
    const auto plus_energy = ecp ? full_energy(plus) : kineticEnergy(private_particles);
    replay->prepareSample(0);
    adapter.installSnapshot(*evaluator, private_particles, minus);
    CHECK(rows.result()[0][parameter].real() == Catch::Approx((plus_log - evaluator->getLogPsi()) / 2e-5).margin(2e-8));
    if (active) CHECK(rows.result()[2][parameter].real() ==
        Catch::Approx(((plus_kinetic - kineticEnergy(private_particles)) / 2e-5).real()).epsilon(2e-6).margin(2e-8));
    const auto minus_energy = ecp ? full_energy(minus) : kineticEnergy(private_particles);
    CHECK(rows.result()[1][parameter].real() == Catch::Approx(((plus_energy - minus_energy) / 2e-5).real()).epsilon(2e-6).margin(2e-8));
  }
  source->evaluateLog(particles);
  CHECK(source->getLogPsi() == original_log);
  CHECK(kineticEnergy(particles) == original_energy);
  sibling->evaluateLog(sibling_particles);
  CHECK(sibling->getLogPsi() == sibling_log);
  auto expected_rng = rng_after_capture;
  CHECK(rng() == expected_rng());
  for (int electron = 0; electron < particles.getTotalNum(); ++electron) CHECK(particles.R[electron] == source_positions[electron]);
  for (int ion = 0; ion < ions.getTotalNum(); ++ion) CHECK(ions.R[ion] == ion_positions[ion]);
  if (active)
  {
    CHECK(rotation(*source).fullRotationCoordinates() == accepted_frame);
    CHECK(rotation(*sibling).fullRotationCoordinates() == accepted_frame);
    for (std::size_t i = 0; i < accepted_coefficients.size(); ++i)
    {
      CHECK(lcao.C->data()[i] == accepted_coefficients.data()[i]);
      CHECK(leaf(*sibling).C->data()[i] == accepted_coefficients.data()[i]);
      CHECK(lcao.originalRotationCoefficients().data()[i] == accepted_reference.data()[i]);
    }
    Libxml2Document unchanged_definition;
    auto unchanged = build(unchanged_definition, xml(asset));
    CHECK(leaf(*unchanged).fileImageIdentity()->image_hex == lcao.fileImageIdentity()->image_hex);
    CHECK(std::filesystem::file_size(asset) == 244008);
    CHECK(nonlocal->captureQuadratureState().coordinates == quadrature.coordinates);
    CHECK(ham.ecpModelProvenance()->scientific_definition == pp_identity);
    CHECK(binding.modelIdentity() == relocated_binding.modelIdentity());
    return; // The tracked HDF image stays read-only; mutation negatives below use disposable copies only.
  }
  Libxml2Document too_small;
  CHECK_THROWS_WITH(build(too_small, xml(asset, 1024)), Catch::Matchers::ContainsSubstring("source_identity_max_bytes"));
  // Capture basis bytes, then change only a disposable file before the actual
  // coefficient open. The old immutable record must reject this second image.
  Libxml2Document changed;
  REQUIRE(changed.parseFromString(xml(temporary)));
  auto* collection = changed.getRoot()->children;
  while (collection && collection->type != XML_ELEMENT_NODE) collection = collection->next;
  LCAOrbitalBuilder builder(particles, ions, OHMMS::Controller, collection);
  const hid_t file = H5Fopen(temporary.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  REQUIRE(file >= 0);
  const hid_t space = H5Screate(H5S_SCALAR);
  const hid_t attribute = H5Acreate2(file, "changed_after_basis", H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
  REQUIRE(attribute >= 0);
  H5Aclose(attribute);
  H5Sclose(space);
  H5Fclose(file);
  auto* spo = collection->children;
  while (spo && spo->type != XML_ELEMENT_NODE) spo = spo->next;
  CHECK_THROWS_WITH(builder.createSPOSetFromXML(spo), Catch::Matchers::ContainsSubstring("changed between actual reader handles"));
  if (ecp)
  {
    Libxml2Document altered_definition;
    auto altered_source = build(altered_definition, xml(temporary));
    TrainingWaveFunctionBinding changed_source(*altered_source, particles, adapterOptions(), *OHMMS::Controller);
    changed_source.setHamiltonian(ham);
    CHECK_THROWS(changed_source.readModel(exported));
  }
  const hid_t linked_file = H5Fopen(temporary.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  REQUIRE(linked_file >= 0);
  REQUIRE(H5Lcreate_external("untracked.h5", "/data", linked_file, "external", H5P_DEFAULT, H5P_DEFAULT) >= 0);
  H5Fclose(linked_file);
  Libxml2Document linked;
  CHECK_THROWS_WITH(build(linked, xml(temporary)), Catch::Matchers::ContainsSubstring("rejects non-hard links"));
  const auto unsupported_dataset = [&](bool virtual_storage) {
    const hid_t file = H5Fopen(temporary.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    REQUIRE(file >= 0);
    if (H5Lexists(file, "external", H5P_DEFAULT) > 0)
      REQUIRE(H5Ldelete(file, "external", H5P_DEFAULT) >= 0);
    if (H5Lexists(file, "unsupported", H5P_DEFAULT) > 0)
      REQUIRE(H5Ldelete(file, "unsupported", H5P_DEFAULT) >= 0);
    const hsize_t extent = 1;
    const hid_t space = H5Screate_simple(1, &extent, nullptr);
    const hid_t properties = H5Pcreate(H5P_DATASET_CREATE);
    if (virtual_storage)
      REQUIRE(H5Pset_virtual(properties, space, "untracked.h5", "/data", space) >= 0);
    else
      REQUIRE(H5Pset_external(properties, "untracked.raw", 0, sizeof(double)) >= 0);
    const hid_t dataset = H5Dcreate2(file, "unsupported", H5T_NATIVE_DOUBLE, space,
                                   H5P_DEFAULT, properties, H5P_DEFAULT);
    REQUIRE(dataset >= 0);
    H5Dclose(dataset);
    H5Pclose(properties);
    H5Sclose(space);
    H5Fclose(file);
    Libxml2Document input;
    CHECK_THROWS_WITH(build(input, xml(temporary)), Catch::Matchers::ContainsSubstring("external storage and virtual datasets"));
  };
  unsupported_dataset(false);
  unsupported_dataset(true);
  // Qualification of one recognized collection never authorizes unrelated
  // paths elsewhere in the model XML.
  // storeXMLNode deep-copies the builder input; mutate the retained definition
  // that Binding actually reads, not the disposable Libxml2Document.
  xmlSetProp(source->getNode(), BAD_CAST "filename", BAD_CAST "unqualified.h5");
  TrainingWaveFunctionBinding unrelated(*source, particles, adapterOptions(), *OHMMS::Controller);
  unrelated.setHamiltonian(ham);
  CHECK_THROWS_WITH(unrelated.modelIdentity(), Catch::Matchers::ContainsSubstring("external"));
}
#endif

} // namespace qmcplusplus::wftrain
