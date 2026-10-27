//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file test_psiformer_complex_wf.cpp
 * @brief Scalar component and XML/device admission regression tests.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#include "Particle/ParticleSet.h"
#include "Particle/ParticleIO/XMLParticleIO.h"
#include "Particle/MCMultiParticleMoves.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerModelFile.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCWaveFunctions/Optimization/NonLocalECPDerivative.h"
#include "QMCHamiltonians/BareKineticEnergy.h"
#include "QMCHamiltonians/ECPComponentBuilder.h"
#include "QMCHamiltonians/NonLocalECPotential.h"
#include "QMCHamiltonians/HarmonicExternalPotential.h"
#include "Utilities/FakeRandom.h"
#include "QMCDrivers/WFTrain/PreparedEnergyGradientProducer.h"
#include "QMCDrivers/WFTrain/NonLocalEnergyAccumulator.h"
#include "QMCDrivers/WFTrain/EnergyGradientAccumulator.h"
#include "QMCDrivers/WFTrain/KFACPreconditioner.h"
#include "QMCDrivers/WFTrain/DistributedParameterReduction.h"
#include "QMCDrivers/WFTrain/MatrixFreeStochasticReconfiguration.h"
#include "QMCDrivers/WFTrain/TrainingCheckpoint.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOrbitalTargets.h"
#include "QMCWaveFunctions/ElectronGas/FreeOrbital.h"
#include "QMCWaveFunctions/Fermion/SlaterDet.h"
#include "QMCWaveFunctions/Fermion/DiracDeterminant.h"
#include "Utilities/RuntimeOptions.h"
#include "OhmmsData/Libxml2Doc.h"
#include "io/hdf/hdf_archive.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWaveFunctionBuilder.h"
#include "QMCHamiltonians/HamiltonianFactory.h"
#include "Particle/VirtualParticleSet.h"
#include "Particle/VirtualParticleBatch.h"
#include "Utilities/ResourceCollection.h"
#include "Utilities/TimerManager.h"
#include "psiformer_test_utils.h"
#ifdef ENABLE_CUDA
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceComplexSpatial.h"
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceComplexProducts.h"
#endif
#include <algorithm>
#include <atomic>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iomanip>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <system_error>
#include <tuple>
#include <thread>
#include <vector>

namespace qmcplusplus
{
namespace testing
{
/// Reuse the existing NLPP test friend without exposing a new production control.
class TestNonLocalECPotential
{
public:
  static void setTileCapacity(NonLocalECPotential& potential, std::size_t capacity)
  { potential.setOuterTileCapacityForTesting(capacity); }
  static bool sameSchema(const NonLocalECPotential& a, const NonLocalECPotential& b)
  { return a.multiWalkerResourceSchema() == b.multiWalkerResourceSchema(); }
};
}
namespace
{
namespace ps = psiformer;
namespace fixture = testing::psiformer;
using C = std::complex<double>;
constexpr std::size_t clone_cap = 8 * 1024 * 1024;

#ifdef ENABLE_CUDA
/// Keep per-test process controls isolated, including restoration of preexisting developer settings.
class ScopedComplexGpuControls
{
public:
  /// Save and clear only the explicitly consumed paired inference controls.
  ScopedComplexGpuControls()
  {
    for (const auto* name : names)
    {
      const char* value = std::getenv(name);
      saved.emplace_back(value ? std::optional<std::string>(value) : std::nullopt);
      ::unsetenv(name);
    }
  }

  /// Restore the caller's environment even when a Catch assertion aborts a section.
  ~ScopedComplexGpuControls()
  {
    for (std::size_t i = 0; i < saved.size(); ++i)
      if (saved[i])
        ::setenv(names[i], saved[i]->c_str(), 1);
      else
        ::unsetenv(names[i]);
  }

  /// Set one strict decimal or policy spelling before constructing a shared model.
  void set(const char* name, const std::string& value) { REQUIRE(::setenv(name, value.c_str(), 1) == 0); }

  /// Select the exact target profile without invoking the legacy REAL automatic planner.
  void target(std::size_t k, std::size_t bytes, std::size_t work)
  {
    set("PSIFORMER_GPU_VGL_TARGETS", std::to_string(k));
    set("PSIFORMER_GPU_TARGET_WORKSPACE_BYTES", std::to_string(bytes));
    set("PSIFORMER_GPU_TARGET_SCALAR_WORK_BUDGET", std::to_string(work));
  }

  const std::vector<const char*> names{"PSIFORMER_GPU_VGL_TARGETS", "PSIFORMER_GPU_TARGET_WORKSPACE_BYTES",
      "PSIFORMER_GPU_TARGET_SCALAR_WORK_BUDGET", "PSIFORMER_GPU_TARGET_SELECTION", "PSIFORMER_GPU_KINETIC_TARGETS",
      "PSIFORMER_GPU_LIFTED_CHANNELS", "PSIFORMER_GPU_RETAINED_COORDINATE_BYTES", "PSIFORMER_GPU_METROPOLIS_BYTES",
      "PSIFORMER_GPU_METROPOLIS_DRIFT_BYTES", "PSIFORMER_GPU_ADAM_BYTES", "PSIFORMER_GPU_RESIDENT_ENERGY_ADAM_BYTES",
      "PSIFORMER_GPU_RESIDENT_ENERGY_SR_BYTES", "PSIFORMER_GPU_RESIDENT_ENERGY_SR_HOST_BYTES", "PSIFORMER_GPU_RESIDENT_ENERGY_SR_REUSE",
      "PSIFORMER_GPU_COMPLEX_TILE_CAPACITY", "PSIFORMER_GPU_ACTIVE_INFERENCE_CROWDS",
      "PSIFORMER_GPU_COMPLEX_VIRTUAL_TILE_CAPACITY", "PSIFORMER_GPU_KFAC_AFFINE",
      "PSIFORMER_GPU_SOFTMAX_KEY_SHARED_MEANS", "PSIFORMER_GPU_COMPLEX_VIRTUAL_ENVELOPE_REUSE"};
  std::vector<std::optional<std::string>> saved;
};
#endif

/// Count any forbidden device-readiness inspection made by profile admission.
class AdmissionReadiness final : public PsiFormerAcceleratorReadinessProvider
{
public:
  /// A rejected GPU request must never reach this seam.
  psiformer::PsiFormerAcceleratorReadiness inspect() const override { ++calls; return {}; }
  mutable int calls = 0;
};

/// Fixed production initializer profile; omitted GPU/optimizer/precision use their CPU-safe defaults.
std::string complexXml(const std::string& extra = "", const std::string& cap = "16777216")
{
  return "<psiformer source='ion0' system='all_electron' feature_policy='periodic_torus_v1' "
         "initialization='psiformer_complex_periodic_v1' initialization_seed='17' " +
         (cap.empty() ? "" : "max_clone_numeric_bytes='" + cap + "' ") + extra + "/>";
}

/// The real production periodic Hamiltonian, not a molecular inverse-distance substitute.
const std::string coulombTerms =
    "<pairpot name='ee' type='coulomb' source='e' target='e' gpu='no'/>"
    "<pairpot name='ei' type='coulomb' source='ion0' target='e'/>"
    "<pairpot name='ii' type='coulomb' source='ion0' target='ion0' gpu='no'/>";

/// Small descriptor fixture exercises the same numerical owners without another Native TU.
/// This is not a new public XML architecture; initializer production dimensions stay fixed.
ps::InitializedPsiFormerParameters parameters(bool paired = true, bool zero_imaginary = false,
                                             std::size_t channels_per_nucleus = 7, std::size_t electron_count = 4)
{
  auto leaves = fixture::makeLayout(electron_count, 2, true, 2, channels_per_nucleus);
  // Keep this compact fixture at D2 even when the head's electron axis is Ne64.
  for (auto& leaf : leaves)
    if (leaf.module.find("exponential_envelopes") != std::string::npos)
      leaf.shape[0] = 2 * electron_count;
    else if (leaf.module.find("Backflow") != std::string::npos)
      leaf.shape.back() = 2 * electron_count;
  for (auto& leaf : leaves)
    for (auto& extent : leaf.shape)
      if (extent == 256)
        extent = 8;
      else if (extent == 64)
        extent = 8;
  if (paired)
  {
    const auto original = leaves;
    for (const auto& leaf : original)
      if (leaf.module.find("Backflow") != std::string::npos)
        leaves.push_back({leaf.module, "w_imag", leaf.shape});
  }
  std::sort(leaves.begin(), leaves.end(), [](const auto& a, const auto& b) {
    return std::tie(a.module, a.name) < std::tie(b.module, b.name);
  });
  ps::InitializedPsiFormerParameters result;
  result.model_shape = {electron_count / 2, electron_count - electron_count / 2, 2, 2, 8, 2, 2};
  result.profile = paired ? ps::PSIFORMER_COMPLEX_PERIODIC_V1 : ps::DEEPQMC_PSIFORMER_V1;
  result.seed = 17;
  // Imaginary leaves must not shift the real stream or be proportional to real heads.
  auto real_leaves = leaves;
  real_leaves.erase(std::remove_if(real_leaves.begin(), real_leaves.end(),
      [](const auto& leaf) { return leaf.name == "w_imag"; }), real_leaves.end());
  const auto real_values = fixture::makeParameters("lih", real_leaves);
  std::size_t real_cursor = 0;
  fixture::SplitMix64 imaginary_random(fixture::PARAMETER_RECIPE_SEED + 199);
  for (const auto& leaf : leaves)
  {
    const std::size_t begin = result.values.size();
    std::vector<double> values(fixture::product(leaf.shape));
    if (leaf.name == "w_imag")
      for (double& value : values)
        value = zero_imaginary ? 0 : .06 * imaginary_random.symmetric();
    else
    {
      std::copy_n(real_values.data() + real_cursor, values.size(), values.data());
      real_cursor += values.size();
    }
    result.values.insert(result.values.end(), values.begin(), values.end());
    result.layouts.push_back({leaf.module, leaf.name, leaf.shape, begin, result.values.size()});
  }
  return result;
}

/// Prepare a skew fully periodic cell, retaining the existing torus feature convention.
Lattice cell()
{
  Lattice lattice;
  lattice.R = {8, 0, 0, .6, 7.4, 0, -.3, .5, 8.5};
  lattice.BoxBConds = {true, true, true};
  lattice.reset();
  return lattice;
}

/// Construct the compact all-electron LiH electron set from the existing fixture recipe.
ParticleSet electrons(const SimulationCell& simulation_cell, int up = 2, int down = 2)
{
  const auto geometry = fixture::makeGeometry("lih");
  ParticleSet result(simulation_cell);
  result.setName("e");
  result.create({up, down});
  auto& species = result.getSpeciesSet();
  species.addSpecies("u");
  species.addSpecies("d");
  const int mass = species.addAttribute("mass");
  species(mass, 0) = species(mass, 1) = 1;
  result.resetGroups();
  for (int electron = 0; electron < 4; ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      result.R[electron][dimension] = geometry.electrons[3 * electron + dimension];
  result.update();
  return result;
}

/// Keep physical source charges/valence/atomic number consistent with all-electron LiH.
ParticleSet ions(const SimulationCell& simulation_cell)
{
  const auto geometry = fixture::makeGeometry("lih");
  ParticleSet result(simulation_cell);
  result.setName("ion0");
  result.create({1, 1});
  auto& species = result.getSpeciesSet();
  species.addSpecies("Li");
  species.addSpecies("H");
  for (const auto* name : {"charge", "valence", "atomicnumber"})
  {
    const int attribute = species.addAttribute(name);
    species(attribute, 0) = 3;
    species(attribute, 1) = 1;
  }
  result.resetGroups();
  for (int atom = 0; atom < 2; ++atom)
    for (int dimension = 0; dimension < 3; ++dimension)
      result.R[atom][dimension] = geometry.nuclei[3 * atom + dimension];
  result.update();
  return result;
}

/// Compare full complex quantities without suppressing their phase components.
void close(C actual, C expected, double tolerance = 2e-10)
{
  CHECK(actual.real() == Catch::Approx(expected.real()).margin(tolerance));
  CHECK(actual.imag() == Catch::Approx(expected.imag()).margin(tolerance));
}

/// Clear only the test's external total products before a fresh full evaluation.
WaveFunctionComponent::LogValue evaluate(WaveFunctionComponent& component, ParticleSet& p)
{
  p.G = 0;
  p.L = 0;
  return component.evaluateLog(p, p.G, p.L);
}
} // namespace

TEST_CASE("Complex orbital targets and a capped CPU MSE adapter preserve fixed gauge and ownership",
          "[wavefunction][psiformer][complex_wf][complex_orbital_adapter]")
{
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell), clone_p = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  const ParticleSet::PosType reduced{.17, -.11, .07};
  p.setTwist(reduced);
  clone_p.setTwist(reduced);
  const std::array<double, 3> twist{reduced[0], reduced[1], reduced[2]};
  const auto cartesian = p.getLattice().k_cart(reduced);
  std::array<ParticleSet::PosType, 2> reciprocal{
      p.getLattice().k_cart(ParticleSet::PosType{1, 0, 0}),
      p.getLattice().k_cart(ParticleSet::PosType{0, 1, 0})};
  std::vector<std::unique_ptr<SPOSet>> spos;
  std::vector<std::unique_ptr<DiracDeterminantBase>> dets;
  for (int spin = 0; spin < 2; ++spin)
  {
    spos.push_back(std::make_unique<FreeOrbital>("target" + std::to_string(spin),
        std::vector<ParticleSet::PosType>{cartesian, cartesian + reciprocal[spin]}));
    dets.push_back(std::make_unique<DiracDeterminant<>>(*spos.back(), 2 * spin, 2 * spin + 2));
  }
  SlaterDet teacher(p, std::move(spos), std::move(dets));
  constexpr auto target_cap = 36 * sizeof(C); // D*N*N plus both occupied SPO scratch vectors.
  CHECK_THROWS(ps::ComplexOrbitalTargetEvaluator::fromSingleReference(teacher, p, twist, 0, target_cap));
  CHECK_THROWS(ps::ComplexOrbitalTargetEvaluator::fromSingleReference(
      teacher, p, twist, std::numeric_limits<std::size_t>::max(), target_cap));
  CHECK_THROWS(ps::ComplexOrbitalTargetEvaluator::fromSingleReference(teacher, p, {}, 2, target_cap));
  CHECK_THROWS(ps::ComplexOrbitalTargetEvaluator::fromSingleReference(teacher, p, twist, 2, target_cap - 1));
  CHECK_THROWS(ps::ComplexOrbitalTargetEvaluator::fromSingleReference(teacher, p, twist, 2, 0));
  CHECK_THROWS(ps::ComplexOrbitalTargetEvaluator::fromSingleReference(
      teacher, p, twist, 2, std::numeric_limits<std::size_t>::max()));
  auto target = ps::ComplexOrbitalTargetEvaluator::fromSingleReference(teacher, p, twist, 2, target_cap);
  const auto original_positions = p.R;
  const auto fixed = target.evaluate(p);
  REQUIRE(fixed.size() == 32);
  for (std::size_t d = 0; d < 2; ++d)
    for (std::size_t row = 0; row < 4; ++row)
      for (std::size_t column = 0; column < 4; ++column)
      {
        const auto index = (d * 4 + row) * 4 + column;
        if ((row < 2) != (column < 2))
          CHECK(fixed[index] == C{});
        else
        {
          double phase = 0;
          if (column % 2)
            for (int dim = 0; dim < 3; ++dim)
              phase += reciprocal[row < 2 ? 0 : 1][dim] * p.R[row][dim];
          close(fixed[index], std::exp(C{0, phase}));
        }
      }
  const auto retained = target.retainedScalarBytes();
  CHECK(retained == target_cap);
  const auto fingerprint = target.descriptor().fingerprint();
  for (int dim = 0; dim < 3; ++dim)
    p.R[0][dim] += p.getLattice().R(0, dim);
  const auto translated = target.evaluate(p);
  for (std::size_t index = 0; index < fixed.size(); ++index)
    close(translated[index], fixed[index]);
  p.R = original_positions;
  CHECK(target.retainedScalarBytes() == retained);
  CHECK(target.descriptor().fingerprint() == fingerprint);
  auto twin_target = ps::ComplexOrbitalTargetEvaluator::fromSingleReference(teacher, p, twist, 2, target_cap);
  CHECK(twin_target.descriptor().fingerprint() == fingerprint);
  // Changing the occupied reciprocal label changes immutable target identity.
  std::vector<std::unique_ptr<SPOSet>> changed_spos;
  std::vector<std::unique_ptr<DiracDeterminantBase>> changed_dets;
  for (int spin = 0; spin < 2; ++spin)
  {
    changed_spos.push_back(std::make_unique<FreeOrbital>("changed" + std::to_string(spin),
        std::vector<ParticleSet::PosType>{cartesian, cartesian + 2.0 * reciprocal[spin]}));
    changed_dets.push_back(std::make_unique<DiracDeterminant<>>(*changed_spos.back(), 2 * spin, 2 * spin + 2));
  }
  SlaterDet changed_teacher(p, std::move(changed_spos), std::move(changed_dets));
  auto changed_target = ps::ComplexOrbitalTargetEvaluator::fromSingleReference(changed_teacher, p, twist, 2, target_cap);
  CHECK(changed_target.descriptor().fingerprint() != fingerprint);
  p.setTwist(ParticleSet::PosType{});
  CHECK_THROWS(target.evaluate(p));
  p.setTwist(reduced);
  auto altered_lattice = cell();
  altered_lattice.R(0, 0) += .1;
  altered_lattice.reset();
  const SimulationCell altered_cell(altered_lattice);
  auto altered_p = electrons(altered_cell);
  altered_p.setTwist(reduced);
  CHECK_THROWS(target.evaluate(altered_p));

  PsiFormerComplexWF component("orbital", parameters(), p, source, clone_cap, twist);
  CHECK(component.reducedTwist() == twist);
  evaluate(component, p);
  const auto accepted = component.get_log_value();
  const auto snapshot = component.snapshotParameters();
  const auto numeric = component.numericStorage();
  std::vector<C> raw_values;
  component.evaluateOrbitalValues(p, raw_values, snapshot.version);
  REQUIRE(raw_values.size() == fixed.size());
  CHECK(component.orbitalMSEStorage().prepared_bytes == 0);
  double value_only_loss = 0;
  for (std::size_t i = 0; i < raw_values.size(); ++i)
    value_only_loss += std::norm(raw_values[i] - fixed[i]) / (component.orbitalMSEChannelCount() * 2 * 4);
  const auto preserved_values = raw_values;
  CHECK_THROWS(component.evaluateOrbitalValues(p, raw_values, snapshot.version + 1));
  CHECK(raw_values == preserved_values);
  CHECK_THROWS(component.evaluateOrbitalValues(altered_p, raw_values, snapshot.version));
  CHECK(raw_values == preserved_values);
  CHECK(component.numericStorage().allocated_bytes == numeric.allocated_bytes);
  CHECK(component.numericStorage().fingerprint == numeric.fingerprint);
  CHECK(component.get_log_value() == accepted);
  std::vector<double> gradient(snapshot.values.size(), 123);
  CHECK_THROWS(component.evaluateOrbitalMSE(p, fixed, gradient, snapshot.version));
  auto owner = component.orbitalMSEStorage();
  REQUIRE(owner.required_bytes > 0);
  CHECK(owner.prepared_bytes == 0);
  CHECK_THROWS(component.prepareOrbitalMSE(0));
  CHECK_THROWS(component.prepareOrbitalMSE(owner.required_bytes - 1));
  CHECK(component.orbitalMSEStorage().prepared_bytes == 0);
  component.prepareOrbitalMSE(owner.required_bytes);
  owner = component.orbitalMSEStorage();
  CHECK(owner.prepared_bytes == owner.required_bytes);
  const double loss = component.evaluateOrbitalMSE(p, fixed, gradient, snapshot.version);
  std::vector<double> full_gradient(gradient.size());
  const double full_loss = component.evaluateOrbitalMSE(p, fixed, full_gradient, snapshot.version,
      wftrain::OrbitalLossNormalization::FULL_MATRIX_MEAN);
  CHECK(full_loss == Catch::Approx(.5 * loss).epsilon(1e-12));
  for (std::size_t index = 0; index < gradient.size(); ++index)
    CHECK(full_gradient[index] == Catch::Approx(.5 * gradient[index]).margin(1e-12));
  CHECK(loss == Catch::Approx(value_only_loss).epsilon(1e-12));
  CHECK(std::isfinite(loss));
  CHECK(std::any_of(gradient.begin(), gradient.end(), [](double value) { return std::abs(value) > 1e-8; }));
  const auto analytic = gradient;
  CHECK(component.evaluateOrbitalMSE(p, fixed, gradient, snapshot.version) == loss);
  CHECK(gradient == analytic);
  CHECK(component.orbitalMSEStorage().storage_fingerprint == owner.storage_fingerprint);
  CHECK(component.get_log_value() == accepted);
  for (int row = 0; row < 4; ++row)
    CHECK(p.R[row] == original_positions[row]);
  auto clone = component.makeClone(clone_p);
  auto& cloned = dynamic_cast<PsiFormerComplexWF&>(*clone);
  CHECK(cloned.orbitalMSEStorage().prepared_bytes == 0);
  cloned.prepareOrbitalMSE(owner.required_bytes);
  CHECK(cloned.orbitalMSEStorage().storage_fingerprint != owner.storage_fingerprint);
  std::fill(gradient.begin(), gradient.end(), 123);
  CHECK_THROWS(component.evaluateOrbitalMSE(p, fixed, gradient, snapshot.version + 1));
  CHECK(std::all_of(gradient.begin(), gradient.end(), [](double value) { return value == 123; }));
  p.makeMove(0, ParticleSet::PosType{.01, 0, 0});
  CHECK_THROWS(component.evaluateOrbitalMSE(p, fixed, gradient, snapshot.version));
  p.rejectMove(0);
  CHECK_THROWS(component.evaluateOrbitalMSE(altered_p, fixed, gradient, snapshot.version));
  auto malformed = fixed;
  malformed.pop_back();
  CHECK_THROWS(component.evaluateOrbitalMSE(p, malformed, gradient, snapshot.version));
  CHECK(std::all_of(gradient.begin(), gradient.end(), [](double value) { return value == 123; }));
  CHECK(component.evaluateOrbitalMSE(p, fixed, gradient, snapshot.version) == loss);
  CHECK(component.snapshotParameters().values == snapshot.values);
  auto candidate = snapshot;
  candidate.values[1] += .01;
  const auto version = component.publishParameters(candidate, snapshot.version);
  CHECK_THROWS(component.evaluateOrbitalMSE(p, fixed, gradient, snapshot.version));
  CHECK(std::isfinite(component.evaluateOrbitalMSE(p, fixed, gradient, version)));
  CHECK(component.orbitalMSEStorage().storage_fingerprint == owner.storage_fingerprint);
}

TEST_CASE("PsiFormer paired neural weights preserve phase head roles and exclude nonweights",
          "[wavefunction][psiformer][complex_wf][complex_weight_blocks]")
{
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell), clone_p = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  const auto initialized = parameters();
  PsiFormerComplexWF component("weights", initialized, p, source, clone_cap, {.17, -.11, .07});
  std::vector<std::string> expected;
  std::size_t imaginary_count = 0;
  for (const auto& leaf : initialized.layouts)
    if (leaf.name == "w" || leaf.name == "w_imag")
    {
      expected.push_back(leaf.module + "/" + leaf.name);
      imaginary_count += leaf.name == "w_imag";
    }
  REQUIRE(expected.size() == 17);
  REQUIRE(imaginary_count == 2);
  const auto weights = component.neuralWeightBlocks();
  CHECK(weights == expected);
  for (const auto& block : component.parameterSchema().blocks())
  {
    const auto leaf = std::find_if(initialized.layouts.begin(), initialized.layouts.end(),
        [&](const auto& item) { return item.module + "/" + item.name == block.id; });
    REQUIRE(leaf != initialized.layouts.end());
    const bool selected = std::find(weights.begin(), weights.end(), block.id) != weights.end();
    CHECK(selected == (leaf->name == "w" || leaf->name == "w_imag"));
    if (selected)
    {
      CHECK(block.offset == leaf->begin);
      CHECK(block.count == leaf->end - leaf->begin);
    }
  }
  evaluate(component, p);
  bool learned_phase_gradient = false;
  for (const auto& gradient : p.G)
    for (int d = 0; d < 3; ++d)
      learned_phase_gradient |= std::abs(gradient[d].imag()) > 1e-8;
  CHECK(learned_phase_gradient);
  auto clone = component.makeClone(clone_p);
  CHECK(dynamic_cast<PsiFormerComplexWF&>(*clone).neuralWeightBlocks() == weights);
  auto candidate = component.snapshotParameters();
  const auto imaginary = std::find_if(initialized.layouts.begin(), initialized.layouts.end(),
      [](const auto& leaf) { return leaf.name == "w_imag"; });
  REQUIRE(imaginary != initialized.layouts.end());
  candidate.values[imaginary->begin] += .01;
  CHECK(component.publishParameters(candidate, candidate.version) == candidate.version + 1);
  CHECK(component.neuralWeightBlocks() == weights);
  CHECK(dynamic_cast<PsiFormerComplexWF&>(*clone).neuralWeightBlocks() == weights);
}

TEST_CASE("PsiFormer standalone paired model preserves complex inference without configuration",
          "[wavefunction][psiformer][complex_wf][model_file]")
{
  const SimulationCell simulation_cell(cell());
  auto particles = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  const std::array<double, 3> twist{.125, -.25, .0625};
  PsiFormerComplexWF original("psiformer", parameters(), particles, source, clone_cap, twist);
  const auto value = evaluate(original, particles);
  const auto gradient = particles.G;
  const auto laplacian = particles.L;
  const auto path = "psiformer_paired_standalone." + std::to_string(OHMMS::Controller->rank()) + ".h5";
  original.exportModel(path);
  WaveFunctionComponentBuilder::PSetMap sources;
  sources.emplace("ion0", std::make_unique<ParticleSet>(source));
  PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, particles, sources);
  Libxml2Document document;
  REQUIRE(document.parseFromString("<psiformer parameters='" + path +
      "' source='ion0' system='all_electron' feature_policy='periodic_torus_v1' "
      "twist='0.125 -0.25 0.0625' max_clone_numeric_bytes='8388608'/>"));
  auto loaded = builder.buildComponent(document.getRoot());
  auto* paired = dynamic_cast<PsiFormerComplexWF*>(loaded.get());
  REQUIRE(paired != nullptr);
  CHECK(paired->snapshotParameters().values == original.snapshotParameters().values);
  close(evaluate(*paired, particles), value);
  for (int electron = 0; electron < particles.getTotalNum(); ++electron)
  {
    for (int d = 0; d < 3; ++d)
      close(particles.G[electron][d], gradient[electron][d]);
    close(particles.L[electron], laplacian[electron]);
  }
  class Sink final : public wftrain::SampleProductSink
  {
  public:
    C value{};
    void onBegin(const wftrain::DerivativeStreamDescriptor&) override { value = 0; }
    void consume(const wftrain::SampleProductTileConstView& tile) override { value = tile.values()[0]; }
  };
  std::vector<C> direction(original.parameterSchema().parameterCount());
  direction[0] = .3;
  for (const auto& block : original.parameterSchema().blocks())
    if (block.id.find("/w_imag") != std::string::npos)
      direction[block.offset] = .2;
  auto product = [&](PsiFormerComplexWF& component) {
    RefVectorWithLeader<WaveFunctionComponent> components(component);
    components.push_back(component);
    RefVectorWithLeader<ParticleSet> particle_refs(particles);
    particle_refs.push_back(particles);
    auto stream = component.makeBudgetedStreamingDerivativeOperator(components, particle_refs, 0, 0, 7,
        component.streamingDerivativeRequiredBytes(1));
    Sink sink;
    stream->applyScoreJVP({component.parameterSchema(), stream->parameterVersion(),
        {direction.data(), direction.size()}}, sink);
    return sink.value;
  };
  close(product(*paired), product(original));
  Libxml2Document wrong_twist;
  REQUIRE(wrong_twist.parseFromString("<psiformer parameters='" + path +
      "' source='ion0' system='all_electron' feature_policy='periodic_torus_v1' "
      "twist='0 0 0' max_clone_numeric_bytes='8388608'/>"));
  CHECK_THROWS(builder.buildComponent(wrong_twist.getRoot()));
  Libxml2Document conflicting_sources;
  REQUIRE(conflicting_sources.parseFromString("<psiformer parameters='" + path +
      "' parameter_checkpoint='missing.root.h5' source='ion0' system='all_electron' "
      "feature_policy='periodic_torus_v1' twist='0.125 -0.25 0.0625' max_clone_numeric_bytes='8388608'/>"));
  CHECK_THROWS_AS(builder.buildComponent(conflicting_sources.getRoot()), std::invalid_argument);
  std::filesystem::remove(path);
}

TEST_CASE("PsiFormer paired system metadata survives private ownership",
          "[wavefunction][psiformer][complex_wf][complex_system_kind]")
{
  const SimulationCell simulation_cell(cell());
  auto particles = electrons(simulation_cell), clone_particles = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  PsiFormerComplexWF implicit_ae("psiformer", parameters(), particles, source, clone_cap);
  PsiFormerComplexWF explicit_ae("psiformer", parameters(), particles, source, clone_cap, {}, 0, {}, "all_electron");
  CHECK(explicit_ae.trainingConfigurationFingerprint() == implicit_ae.trainingConfigurationFingerprint());
  CHECK_FALSE(explicit_ae.isPseudopotentialSystem());
  CHECK_FALSE(explicit_ae.matchesLegacyTorusPseudopotentialTarget(particles));
  CHECK(implicit_ae.matchesLegacyTorusAllElectronTarget(particles));
  CHECK(explicit_ae.matchesLegacyTorusAllElectronTarget(particles));
  const SimulationCell open_cell;
  auto open_particles = electrons(open_cell);
  CHECK_FALSE(implicit_ae.matchesLegacyTorusAllElectronTarget(open_particles));
  CHECK_THROWS_AS(PsiFormerComplexWF("psiformer", parameters(), particles, source, clone_cap,
      {}, 0, {}, "invalid"), std::invalid_argument);
  // Metadata-only roundtrip: this does not admit a native pseudopotential Hamiltonian/profile.
  PsiFormerComplexWF pp("psiformer", parameters(), particles, source, clone_cap, {}, 0, {}, "pseudopotential");
  CHECK(pp.isPseudopotentialSystem());
  CHECK_FALSE(pp.matchesLegacyTorusAllElectronTarget(particles));
  CHECK(pp.matchesLegacyTorusPseudopotentialTarget(particles));
  CHECK_FALSE(pp.matchesLegacyTorusPseudopotentialTarget(open_particles));
  CHECK_FALSE(pp.matchesLegacyTorusPseudopotentialTarget(clone_particles));
  particles.setTwist(ParticleSet::PosType{.125, 0, 0});
  CHECK_FALSE(pp.matchesLegacyTorusPseudopotentialTarget(particles));
  particles.setTwist(ParticleSet::PosType{});
  const auto fingerprint = pp.trainingConfigurationFingerprint();
  CHECK(fingerprint != explicit_ae.trainingConfigurationFingerprint());
  auto clone = pp.makeClone(clone_particles);
  auto* private_pp = dynamic_cast<PsiFormerComplexWF*>(clone.get());
  REQUIRE(private_pp != nullptr);
  private_pp->isolateParameterState();
  CHECK(private_pp->matchesLegacyTorusPseudopotentialTarget(clone_particles));
  CHECK(private_pp->trainingConfigurationFingerprint() == fingerprint);
  const auto path = "psiformer_paired_kind." + std::to_string(OHMMS::Controller->rank()) + ".h5";
  private_pp->exportModel(path);
  CHECK(psiformer::readPsiFormerModelFile(path).system == "pseudopotential");
  std::filesystem::remove(path);
}

#ifndef MIXED_PRECISION
TEST_CASE("PsiFormer paired immediate FULL replay proves fresh physical products and weak identity",
          "[wavefunction][psiformer][complex_wf][immediate_full_replay]")
{
  SimulationCell simulation_cell(cell());
  auto live = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  const std::array<double, 3> twist{.125, -.0625, .03125};
  live.setTwist(ParticleSet::PosType{twist[0], twist[1], twist[2]});
  PsiFormerComplexWF component("replay", parameters(), live, source, clone_cap, twist, 0, {}, "pseudopotential");
  CHECK_FALSE(component.canReuseImmediateFullSpatialReplay(live, live, {}, {},
      component.immediateFullReplayContext(), 0, 0));
  live.G = C{};
  live.L = C{};
  const auto log = component.evaluateLog(live, live.G, live.L);
  ParticleSet saved(live);
  auto context = component.immediateFullReplayContext();
  std::vector<WaveFunctionComponent::ValueType> gradient, laplacian;
  for (int e = 0; e < live.getTotalNum(); ++e)
  {
    for (int d = 0; d < 3; ++d)
      gradient.push_back(live.G[e][d]);
    laplacian.push_back(live.L[e]);
  }
  double expected_log = std::real(log), expected_phase = std::imag(log);
  auto matches = [&] {
    return component.canReuseImmediateFullSpatialReplay(live, saved, gradient, laplacian,
                                                         context, expected_log, expected_phase);
  };
  REQUIRE(matches());
  const auto storage = component.numericStorage();
  bool positive = false;
  SECTION("fresh proof is independent of stale retained derivatives")
  {
    saved.G[0][0] += C{.1, .2};
    saved.L[0] -= C{.3, .4};
    positive = true;
  }
  SECTION("imaginary fresh gradient changed") { gradient[0] += C{0, .125}; }
  SECTION("real fresh Laplacian changed") { laplacian[0] += C{.125, 0}; }
  SECTION("live derivatives changed") { live.G[0][0] += C{.125, 0}; }
  SECTION("fresh extent changed") { gradient.pop_back(); }
  SECTION("live coordinate extent changed") { live.R.resize(1); }
  SECTION("saved spin extent changed") { saved.spins.resize(1); }
  SECTION("nonfinite fresh view") { laplacian[0] = C{0, std::numeric_limits<double>::infinity()}; }
  SECTION("coordinates changed") { live.R[0][0] += .125; }
  SECTION("saved coordinates changed") { saved.R[0][0] += .125; }
  SECTION("spin changed") { saved.spins[0] += .125; }
  SECTION("group changed") { saved.GroupID[0] = 1; }
  SECTION("mass stamp changed") { context.masses[0] += .125; }
  SECTION("source changed") { source.R[0][0] += .125; }
  SECTION("source charge changed")
  {
    auto& species = source.getSpeciesSet();
    species(species.findAttribute("charge"), 0) += .125;
  }
  SECTION("shared cell changed") { simulation_cell.getLattice().R(0, 0) += .125; }
  SECTION("sub-tolerance shared cell change is not exact evidence")
  {
    auto& r = simulation_cell.getLattice().R(0, 0);
    r = std::nextafter(r, std::numeric_limits<double>::infinity());
  }
  SECTION("nonfinite shared cell")
  {
    simulation_cell.getLattice().R(0, 0) = std::numeric_limits<double>::quiet_NaN();
  }
  SECTION("twist changed") { live.setTwist(ParticleSet::PosType{}); }
  SECTION("foreign source stamp") { context.source = &saved; }
  SECTION("stale version stamp") { ++context.version; }
  SECTION("expired model stamp") { context.model.reset(); }
  SECTION("foreign live target")
  {
    ParticleSet foreign(live);
    CHECK_FALSE(component.canReuseImmediateFullSpatialReplay(foreign, saved, gradient, laplacian,
                                                             context, expected_log, expected_phase));
    positive = true;
  }
  SECTION("foreign same-shape model stamp")
  {
    PsiFormerComplexWF foreign("foreign", parameters(), live, source, clone_cap, twist, 0, {}, "pseudopotential");
    context.model = foreign.immediateFullReplayContext().model;
    CHECK_FALSE(matches());
  }
  SECTION("missing device image") { context.device_image = true; }
  SECTION("log changed") { expected_log += .125; }
  SECTION("phase changed") { expected_phase += .125; }
  SECTION("active unresolved proposal") { live.makeMove(0, ParticleSet::PosType{.01, 0, 0}); }
  SECTION("same-family parameter publication invalidates proof")
  {
    const auto original = component.snapshotParameters();
    auto changed = original;
    changed.values[0] += .001;
    component.publishParameters(changed, original.version);
  }
  SECTION("isolated model replacement invalidates weak ownership") { component.isolateParameterState(); }
  CHECK(matches() == positive);
  CHECK(matches() == positive);
  CHECK(component.numericStorage().allocated_bytes == storage.allocated_bytes);
}
#endif

TEST_CASE("PsiFormer standalone native real Gamma remains a real model in a complex build",
          "[wavefunction][psiformer][complex_wf][model_file]")
{
  const SimulationCell simulation_cell(cell());
  auto particles = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  PsiFormerWF original("psiformer", parameters(false), particles, source);
  original.validateSystem(particles, source, "all_electron");
  const auto value = evaluate(original, particles);
  const auto path = "psiformer_real_gamma_standalone." + std::to_string(OHMMS::Controller->rank()) + ".h5";
  original.exportModel(path);
  WaveFunctionComponentBuilder::PSetMap sources;
  sources.emplace("ion0", std::make_unique<ParticleSet>(source));
  PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, particles, sources);
  Libxml2Document document;
  REQUIRE(document.parseFromString("<psiformer parameters='" + path +
      "' source='ion0' system='all_electron' feature_policy='periodic_torus_v1'/>"));
  auto loaded = builder.buildComponent(document.getRoot());
  REQUIRE(dynamic_cast<PsiFormerWF*>(loaded.get()) != nullptr);
  CHECK(dynamic_cast<PsiFormerComplexWF*>(loaded.get()) == nullptr);
  close(evaluate(*loaded, particles), value);
  std::filesystem::remove(path);
}

TEST_CASE("PsiFormer explicit complex scalar sibling specializes to the legacy real component",
          "[wavefunction][psiformer][complex_wf]")
{
  const SimulationCell simulation_cell(cell());
  auto complex_particles = electrons(simulation_cell);
  auto real_particles = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  PsiFormerComplexWF complex("complex", parameters(true, true), complex_particles, source, clone_cap);
  PsiFormerWF real("real", parameters(false), real_particles, source);
  const auto actual = evaluate(complex, complex_particles);
  const auto expected = evaluate(real, real_particles);
  close(std::exp(actual - expected), C{1});
  CHECK(complex.isFermionic());
  CHECK(complex.isMultiDet());
  CHECK_FALSE(complex.isOptimizable());
  CHECK(complex.structuredParameterProvider() == &complex);
  for (int electron = 0; electron < 4; ++electron)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      close(complex_particles.G[electron][dimension], real_particles.G[electron][dimension]);
    close(complex_particles.L[electron], real_particles.L[electron]);
  }
  complex.evaluateLog(complex_particles, complex_particles.G, complex_particles.L);
  for (int electron = 0; electron < 4; ++electron)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      close(complex_particles.G[electron][dimension], 2.0 * real_particles.G[electron][dimension]);
    close(complex_particles.L[electron], 2.0 * real_particles.L[electron]);
  }
}

TEST_CASE("PsiFormer complex scalar sibling applies one outer phase and physical VGL",
          "[wavefunction][psiformer][complex_wf]")
{
  const auto lattice = cell();
  const SimulationCell simulation_cell(lattice);
  auto gamma_particles = electrons(simulation_cell);
  auto twisted_particles = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  const ParticleSet::PosType reduced{.17, -.11, .07};
  const auto k = lattice.k_cart(reduced);
  PsiFormerComplexWF gamma("gamma", parameters(), gamma_particles, source, clone_cap);
  PsiFormerComplexWF twisted("twisted", parameters(), twisted_particles, source, clone_cap, {.17, -.11, .07});
  const auto gamma_log = evaluate(gamma, gamma_particles);
  const auto twisted_log = evaluate(twisted, twisted_particles);
  double angle = 0;
  for (int electron = 0; electron < 4; ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
    {
      angle += k[dimension] * twisted_particles.R[electron][dimension];
      close(twisted_particles.G[electron][dimension] - gamma_particles.G[electron][dimension], C{0, k[dimension]});
    }
  close(std::exp(twisted_log - gamma_log), std::exp(C{0, angle}));
  for (int electron = 0; electron < 4; ++electron)
    close(twisted_particles.L[electron], gamma_particles.L[electron]);
  const auto old_g = twisted_particles.G;
  const auto old_l = twisted_particles.L;
  for (int dimension = 0; dimension < 3; ++dimension)
    twisted_particles.R[0][dimension] += lattice.R(0, dimension);
  twisted_particles.update();
  const auto image_log = evaluate(twisted, twisted_particles);
  double image_angle = 0;
  for (int dimension = 0; dimension < 3; ++dimension)
    image_angle += k[dimension] * lattice.R(0, dimension);
  close(std::exp(image_log - twisted_log), std::exp(C{0, image_angle}));
  for (int electron = 0; electron < 4; ++electron)
  {
    for (int dimension = 0; dimension < 3; ++dimension)
      close(twisted_particles.G[electron][dimension], old_g[electron][dimension]);
    close(twisted_particles.L[electron], old_l[electron]);
  }
  const auto g = twisted_particles.G;
  const auto l = twisted_particles.L;
  constexpr double step = 2e-4;
  for (int electron : {0, 2})
  {
    C lap{};
    const auto active = twisted.evalGrad(twisted_particles, electron);
    for (int dimension = 0; dimension < 3; ++dimension)
    {
      close(active[dimension], g[electron][dimension]);
      ParticleSet::PosType displacement(0);
      displacement[dimension] = step;
      twisted_particles.makeMove(electron, displacement);
      const C plus = twisted.ratio(twisted_particles, electron);
      twisted.restore(electron);
      twisted_particles.rejectMove(electron);
      displacement[dimension] = -step;
      twisted_particles.makeMove(electron, displacement);
      const C minus = twisted.ratio(twisted_particles, electron);
      twisted.restore(electron);
      twisted_particles.rejectMove(electron);
      close(g[electron][dimension], (plus - minus) / (2 * step), 3e-6);
      lap += (plus + minus - C{2}) / (step * step);
    }
    C ratio_laplacian = l[electron];
    for (int dimension = 0; dimension < 3; ++dimension)
      ratio_laplacian += g[electron][dimension] * g[electron][dimension];
    close(ratio_laplacian, lap, 3e-5);
  }
}

TEST_CASE("PsiFormer complex scalar move state commits only matching finite proposals",
          "[wavefunction][psiformer][complex_wf]")
{
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  PsiFormerComplexWF component("moves", parameters(), p, source, clone_cap, {.17, -.11, .07});
  const ParticleSet::PosType displacement{.02, -.01, .03};
  p.makeMove(0, displacement);
  CHECK_THROWS(component.ratio(p, 0));
  CHECK_THROWS(component.acceptMove(p, 0));
  p.rejectMove(0);
  evaluate(component, p);
  for (int electron : {0, 2})
  {
    const auto before = component.get_log_value();
    p.makeMove(electron, displacement);
    WaveFunctionComponent::GradType gradient(C{.1, -.2});
    const C ratio = component.ratioGrad(p, electron, gradient);
    component.acceptMove(p, electron);
    p.acceptMove(electron);
    const auto cached = component.get_log_value();
    close(ratio, std::exp(cached - before));
    const auto refreshed = evaluate(component, p);
    close(std::exp(cached - refreshed), C{1});
    for (int dimension = 0; dimension < 3; ++dimension)
      close(gradient[dimension], p.G[electron][dimension] + C{.1, -.2});
    p.makeMove(electron, displacement);
    component.ratio(p, electron);
    component.restore(electron);
    p.rejectMove(electron);
    close(component.get_log_value(), refreshed);
  }
  p.makeMove(0, displacement);
  CHECK_THROWS(component.ratio(p, 1));
  CHECK_THROWS(component.acceptMove(p, 0));
  p.rejectMove(0);
  // A proposed-key check alone cannot see a mutated old position of the moved row.
  p.makeMove(0, displacement);
  component.ratio(p, 0);
  const auto reference = p.R[0];
  p.R[0][0] += .125;
  CHECK_THROWS(component.acceptMove(p, 0));
  p.R[0] = reference;
  CHECK_THROWS(component.acceptMove(p, 0)); // failed acceptance revoked the proposal
  p.rejectMove(0);
  p.makeMove(0, displacement);
  component.ratio(p, 0);
  CHECK_THROWS(component.acceptMove(p, 1));
  CHECK_THROWS(component.acceptMove(p, 0));
  p.rejectMove(0);
  evaluate(component, p); // ordinary full refresh recovers after both provenance failures
  // Exact same-spin nodes admit a zero VALUE ratio, never undefined ACTIVE jets.
  // Dyadic inputs make the active replacement exact, not a rounded near-node.
  p.R[0] = {1, 1, 1};
  p.R[1] = {1.5, 1.25, .75};
  p.update();
  evaluate(component, p);
  p.makeMove(0, p.R[1] - p.R[0]);
  close(component.ratio(p, 0), C{});
  component.restore(0);
  WaveFunctionComponent::GradType untouched(C{.1, -.2});
  CHECK_THROWS(component.ratioGrad(p, 0, untouched));
  for (int dimension = 0; dimension < 3; ++dimension)
    close(untouched[dimension], C{.1, -.2});
  p.rejectMove(0);
  CHECK(component.makeClone(p) != nullptr);
}

TEST_CASE("PsiFormer complex clone admits exact numeric bytes and independent scratch",
          "[wavefunction][psiformer][complex_wf]")
{
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell);
  auto clone_particles = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  CHECK_THROWS_AS(PsiFormerComplexWF("zero", parameters(), p, source, 0), std::invalid_argument);
  CHECK_THROWS_AS(PsiFormerComplexWF("unlimited", parameters(), p, source,
      std::numeric_limits<std::size_t>::max()), std::invalid_argument);
  PsiFormerComplexWF component("source", parameters(), p, source, clone_cap, {.17, -.11, .07});
  const auto storage = component.numericStorage();
  REQUIRE(storage.required_bytes > 0);
  CHECK(storage.required_bytes == storage.allocated_bytes);
  CHECK(storage.limit_bytes == clone_cap);
  CHECK_THROWS_AS(PsiFormerComplexWF("short", parameters(), p, source, storage.required_bytes - 1),
                  std::length_error);
  PsiFormerComplexWF exact("exact", parameters(), p, source, storage.required_bytes);
  CHECK(exact.numericStorage().allocated_bytes == storage.required_bytes);

  const auto original_log = evaluate(component, p);
  const auto original_g = p.G;
  p.makeMove(0, ParticleSet::PosType{.031, -.021, .012});
  const auto original_ratio = component.ratio(p, 0);
  auto cloned = component.makeClone(clone_particles);
  auto* clone = dynamic_cast<PsiFormerComplexWF*>(cloned.get());
  REQUIRE(clone != nullptr);
  CHECK(&clone->targetParticles() == &clone_particles);
  CHECK(&component.targetParticles() == &p);
  CHECK(clone->getName() == component.getName());
  const auto clone_storage = clone->numericStorage();
  CHECK(clone_storage.required_bytes == storage.required_bytes);
  CHECK(clone_storage.allocated_bytes == storage.allocated_bytes);
  CHECK(clone_storage.limit_bytes == clone_cap);
  CHECK(clone_storage.fingerprint != storage.fingerprint);
  clone_particles.makeMove(0, ParticleSet::PosType{.021, .011, -.013});
  CHECK_THROWS(clone->ratio(clone_particles, 0)); // accepted/pending state is never cloned
  clone_particles.rejectMove(0);
  close(std::exp(evaluate(*clone, clone_particles) - original_log), C{1});
  for (int electron = 0; electron < 4; ++electron)
    for (int dimension = 0; dimension < 3; ++dimension)
      close(clone_particles.G[electron][dimension], original_g[electron][dimension]);

  // Evaluating and moving the clone cannot invalidate the source's pending proposal.
  clone_particles.makeMove(2, ParticleSet::PosType{-.017, .025, .014});
  WaveFunctionComponent::GradType gradient{};
  clone->ratioGrad(clone_particles, 2, gradient);
  clone->acceptMove(clone_particles, 2);
  clone_particles.acceptMove(2);
  const auto clone_log = clone->get_log_value();
  close(std::exp(evaluate(*clone, clone_particles) - clone_log), C{1});
  component.acceptMove(p, 0);
  p.acceptMove(0);
  close(std::exp(evaluate(component, p) - original_log), original_ratio);
  CHECK_THROWS(component.evaluateLog(clone_particles, clone_particles.G, clone_particles.L));
  evaluate(component, p); // a foreign-target failure revokes publication and remains recoverable
  for (int repeat = 0; repeat < 2; ++repeat)
  {
    evaluate(component, p);
    component.evalGrad(p, 1);
    p.makeMove(3, ParticleSet::PosType{.005, .004, -.003});
    component.ratio(p, 3);
    component.restore(3);
    p.rejectMove(3);
  }
  CHECK(component.numericStorage().allocated_bytes == storage.allocated_bytes);
  CHECK(component.numericStorage().fingerprint == storage.fingerprint);
  CHECK(clone->numericStorage().fingerprint == clone_storage.fingerprint);
}

TEST_CASE("Complex CPU shared scratch is lazy, capped and independent of walker population",
          "[wavefunction][psiformer][complex_wf][complex_cpu_shared]")
{
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  const std::array<double, 3> twist{.17, -.11, .07};
  PsiFormerComplexWF shared("shared", parameters(), p, source, clone_cap, twist, 0, {}, "all_electron", clone_cap);
  const auto cold = shared.numericStorage();
  CHECK(cold.allocated_bytes == 4 * (sizeof(WaveFunctionComponent::GradType) + sizeof(WaveFunctionComponent::ValueType)));
  CHECK(cold.allocated_bytes == cold.required_bytes);
  CHECK(cold.shared_cpu_bytes == 0);
  CHECK(cold.shared_cpu_owners == 1);
  REQUIRE(cold.shared_cpu_identity != nullptr);
  std::vector<std::unique_ptr<WaveFunctionComponent>> population;
  for (int i = 1; i < 4096; ++i)
    population.push_back(shared.makeClone(p));
  const auto& last = dynamic_cast<PsiFormerComplexWF&>(*population.back());
  CHECK(last.numericStorage().shared_cpu_identity == cold.shared_cpu_identity);
  CHECK(last.numericStorage().allocated_bytes == cold.allocated_bytes);
  CHECK(shared.numericStorage().shared_cpu_owners == 1);
  CHECK(shared.numericStorage().shared_cpu_bytes == 0);

  std::vector<C> orbitals;
  shared.evaluateOrbitalValues(p, orbitals, shared.snapshotParameters().version);
  CHECK(shared.numericStorage().shared_cpu_bytes == cold.shared_cpu_value_bytes);
  PsiFormerComplexWF legacy("legacy", parameters(), p, source, clone_cap, twist);
  const auto reference = evaluate(legacy, p);
  const auto expected_g = p.G;
  const auto expected_l = p.L;
  close(evaluate(shared, p), reference, 2e-12);
  CHECK(shared.numericStorage().shared_cpu_bytes == cold.shared_cpu_value_bytes + cold.shared_cpu_full_bytes);
  for (int e = 0; e < 4; ++e)
  {
    for (int d = 0; d < 3; ++d)
      close(p.G[e][d], expected_g[e][d], 2e-12);
    close(p.L[e], expected_l[e], 2e-12);
  }
  p.makeMove(0, ParticleSet::PosType{.01, -.02, .03});
  WaveFunctionComponent::GradType gradient{};
  shared.ratioGrad(p, 0, gradient);
  shared.restore(0);
  p.rejectMove(0);
  const auto warm = shared.numericStorage();
  CHECK(warm.shared_cpu_bytes == cold.shared_cpu_value_bytes + cold.shared_cpu_full_bytes + cold.shared_cpu_active_bytes);
  for (int repeat = 0; repeat < 3; ++repeat)
    close(evaluate(shared, p), reference, 2e-12);
  CHECK(shared.numericStorage().shared_cpu_bytes == warm.shared_cpu_bytes);
  CHECK(shared.numericStorage().shared_cpu_fingerprint == warm.shared_cpu_fingerprint);
  CHECK(shared.numericStorage().fingerprint == cold.fingerprint);

  // A later failed mode must not revoke the already prepared VALUE owner.
  PsiFormerComplexWF short_cap("short", parameters(), p, source, clone_cap, twist, 0, {}, "all_electron",
      cold.shared_cpu_value_bytes + cold.shared_cpu_full_bytes - 1);
  short_cap.evaluateOrbitalValues(p, orbitals, short_cap.snapshotParameters().version);
  CHECK_THROWS_AS(evaluate(short_cap, p), std::length_error);
  CHECK(short_cap.numericStorage().shared_cpu_bytes == cold.shared_cpu_value_bytes);
  CHECK_NOTHROW(short_cap.evaluateOrbitalValues(p, orbitals, short_cap.snapshotParameters().version));
  PsiFormerComplexWF first_short("first_short", parameters(), p, source, clone_cap, twist, 0, {}, "all_electron",
      cold.shared_cpu_full_bytes - 1);
  CHECK_THROWS_AS(evaluate(first_short, p), std::length_error);
  CHECK(first_short.numericStorage().shared_cpu_bytes == 0);
  CHECK(first_short.numericStorage().shared_cpu_fingerprint == 0);
  PsiFormerComplexWF exact("exact", parameters(), p, source, clone_cap, twist, 0, {}, "all_electron",
      cold.shared_cpu_full_bytes);
  close(evaluate(exact, p), reference, 2e-12);
  CHECK(exact.numericStorage().shared_cpu_bytes == cold.shared_cpu_full_bytes);
  CHECK_THROWS_AS(PsiFormerComplexWF("unlimited", parameters(), p, source, clone_cap, twist, 0, {}, "all_electron",
      std::numeric_limits<std::size_t>::max()), std::invalid_argument);
  CHECK_THROWS_AS(PsiFormerComplexWF("mixed", parameters(), p, source, clone_cap, twist, clone_cap, {}, "all_electron",
      clone_cap), std::invalid_argument);
  shared.isolateParameterState();
  CHECK(shared.numericStorage().shared_cpu_identity != cold.shared_cpu_identity);
  CHECK(shared.numericStorage().shared_cpu_limit_bytes == clone_cap);
  CHECK(shared.numericStorage().shared_cpu_bytes == 0);
  close(evaluate(shared, p), reference, 2e-12);
}

TEST_CASE("Complex shared CPU collective proposals retain atomic complete products",
          "[wavefunction][psiformer][complex_wf][complex_cpu_collective]")
{
  const SimulationCell simulation_cell(cell());
  auto p0 = electrons(simulation_cell), p1 = electrons(simulation_cell);
  p1.R[1][2] += .09;
  p1.update();
  auto source = ions(simulation_cell);
  const auto twist = GENERATE(std::array<double, 3>{}, (std::array<double, 3>{.17, -.11, .07}));
  const auto selected = GENERATE(1, 2, 3, 4);
  CAPTURE(selected);
  PsiFormerComplexWF family("collective", parameters(), p0, source, clone_cap, twist, 0, {}, "all_electron", clone_cap);
  auto other = family.makeClone(p1);
  auto& clone = dynamic_cast<PsiFormerComplexWF&>(*other);
  PsiFormerComplexWF private_cpu("private", parameters(), p0, source, clone_cap, twist);
  CHECK_FALSE(private_cpu.supportsMultiParticleMoves());
  REQUIRE(family.supportsMultiParticleMoves());
  const std::array<C, 2> old_logs{evaluate(family, p0), evaluate(clone, p1)};
  const std::array<decltype(p0.R), 2> old_positions{p0.R, p1.R};
  const std::array<ParticleSet::ParticleGradient, 2> old_g{p0.G, p1.G};
  const std::array<ParticleSet::ParticleLaplacian, 2> old_l{p0.L, p1.L};
  RefVectorWithLeader<WaveFunctionComponent> team(family, {family, clone});
  RefVectorWithLeader<ParticleSet> targets(p0, {p0, p1});
  std::vector<ParticleSet::IndexType> indices;
  std::vector<ParticleSet::PosType> positions;
  for (int walker = 0; walker < 2; ++walker)
    for (int row = 0; row < selected; ++row)
    {
      const int electron = selected == 4 ? row : row + 1;
      indices.push_back(electron);
      positions.push_back(old_positions[walker][electron] + ParticleSet::PosType{.003 * (positions.size() + 1), -.002, .004});
    }
  using Moves = MCMultiParticleMoves<CoordsType::POS>;
  const Moves moves({0, std::size_t(selected), std::size_t(2 * selected)}, indices, positions);
  std::array<ParticleSet::ParticleGradient, 2> proposed_g{ParticleSet::ParticleGradient(4), ParticleSet::ParticleGradient(4)};
  std::array<ParticleSet::ParticleLaplacian, 2> proposed_l{ParticleSet::ParticleLaplacian(4), ParticleSet::ParticleLaplacian(4)};
  RefVector<ParticleSet::ParticleGradient> gradients{proposed_g[0], proposed_g[1]};
  RefVector<ParticleSet::ParticleLaplacian> laplacians{proposed_l[0], proposed_l[1]};
  std::array<C, 2> oracle_logs;
  auto oracle_g = proposed_g;
  auto oracle_l = proposed_l;
  for (std::size_t walker = 0; walker < 2; ++walker)
  {
    auto proposed = electrons(simulation_cell);
    proposed.R = old_positions[walker];
    for (int row = 0; row < selected; ++row)
      proposed.R[indices[walker * selected + row]] = positions[walker * selected + row];
    proposed.update();
    auto oracle = private_cpu.makeClone(proposed);
    oracle_logs[walker] = evaluate(*oracle, proposed);
    oracle_g[walker] = proposed.G;
    oracle_l[walker] = proposed.L;
  }
  ResourceCollection prototype("CPU collective" );
  family.createResource(prototype);
  const auto before = family.numericStorage();
  {
    ResourceCollection resource(prototype);
    const auto cursor = resource.getCursor();
    family.acquireResource(resource, team);
    std::vector<C> ratios(2, C{9, 7});
    auto reset = [&] {
      ratios.assign(2, C{9, 7});
      for (std::size_t i = 0; i < 2; ++i)
      {
        proposed_g[i] = C{};
        proposed_l[i] = C{};
      }
    };
    reset();
    auto alias = gradients;
    alias[1] = std::ref(p0.G);
    CHECK_THROWS(family.mw_evaluateMultiParticleMove(team, targets, moves, ratios, alias, laplacians));
    family.mw_evaluateMultiParticleMove(team, targets, moves, ratios, gradients, laplacians);
    const auto warm = family.numericStorage();
    CHECK(warm.shared_cpu_bytes > before.shared_cpu_bytes + before.shared_cpu_full_bytes);
    for (std::size_t i = 0; i < 2; ++i)
    {
      close(ratios[i], oracle_logs[i] - old_logs[i], 2e-12);
      CHECK((i ? clone : family).get_log_value() == old_logs[i]);
      for (std::size_t e = 0; e < 4; ++e)
      {
        close(proposed_l[i][e], oracle_l[i][e], 2e-12);
        CHECK(targets[i].L[e] == old_l[i][e]);
        for (int d = 0; d < 3; ++d)
        {
          close(proposed_g[i][e][d], oracle_g[i][e][d], 2e-12);
          CHECK(targets[i].G[e][d] == old_g[i][e][d]);
        }
      }
    }
    CHECK_THROWS(family.mw_accept_rejectMultiParticleMove(team, targets, moves, std::vector<bool>{false}));
    family.mw_accept_rejectMultiParticleMove(team, targets, moves, std::vector<bool>{false, false});
    reset();
    proposed_l[1][3] = C{std::numeric_limits<double>::infinity(), 0};
    CHECK_THROWS(family.mw_evaluateMultiParticleMove(team, targets, moves, ratios, gradients, laplacians));
    CHECK(ratios[0] == (C{9, 7}));
    CHECK(ratios[1] == (C{9, 7}));
    reset();
    family.mw_evaluateMultiParticleMove(team, targets, moves, ratios, gradients, laplacians);
    CHECK(family.numericStorage().shared_cpu_bytes == warm.shared_cpu_bytes);
    CHECK(family.numericStorage().shared_cpu_fingerprint == warm.shared_cpu_fingerprint);
    for (int row = 0; row < selected; ++row)
      p0.R[indices[row]] = positions[row];
    p0.update();
    family.mw_accept_rejectMultiParticleMove(team, targets, moves, std::vector<bool>{true, false});
    close(family.get_log_value(), oracle_logs[0], 2e-12);
    CHECK(clone.get_log_value() == old_logs[1]);
    for (int electron = 0; electron < 4; ++electron)
      for (int d = 0; d < 3; ++d)
        CHECK(p1.R[electron][d] == old_positions[1][electron][d]);
    for (int row = selected; row < 4; ++row)
      CHECK(p0.R[(row + 1) % 4] == old_positions[0][(row + 1) % 4]);
    CHECK_NOTHROW(family.observeCoulombEnergy(p0, 0));
    CHECK_NOTHROW(clone.observeCoulombEnergy(p1, 0));
    resource.rewind(cursor);
    family.releaseResource(resource, team);
    CHECK(family.numericStorage().shared_cpu_bytes == warm.shared_cpu_bytes);
  }
  CHECK(family.numericStorage().shared_cpu_bytes == before.shared_cpu_bytes);
  CHECK(family.numericStorage().shared_cpu_fingerprint == before.shared_cpu_fingerprint);
}

TEST_CASE("Complex CPU collective products obey exact family admission and growth rollback",
          "[wavefunction][psiformer][complex_wf][complex_cpu_collective]")
{
  const SimulationCell simulation_cell(cell());
  auto p0 = electrons(simulation_cell), p1 = electrons(simulation_cell);
  p1.R[2][0] += .04;
  p1.update();
  auto source = ions(simulation_cell);
  PsiFormerComplexWF descriptor("descriptor", parameters(), p0, source, clone_cap, {}, 0, {}, "all_electron", clone_cap);
  const auto full = descriptor.numericStorage().shared_cpu_full_bytes;
  using Grad = WaveFunctionComponent::GradType;
  using Value = WaveFunctionComponent::ValueType;
  const auto staging = 3 * sizeof(double) + sizeof(C) + sizeof(std::uint64_t) + 4 * (sizeof(Grad) + sizeof(Value));
  const bool first_fails = GENERATE(true, false);
  const auto cap = first_fails ? full + staging - 1 : 2 * full + staging;
  PsiFormerComplexWF family("cap", parameters(), p0, source, clone_cap, {}, 0, {}, "all_electron", cap);
  auto other = family.makeClone(p1);
  auto& clone = dynamic_cast<PsiFormerComplexWF&>(*other);
  evaluate(family, p0);
  evaluate(clone, p1);
  const auto before = family.numericStorage();
  RefVectorWithLeader<WaveFunctionComponent> one(family, {family});
  RefVectorWithLeader<ParticleSet> target(p0, {p0});
  const MCMultiParticleMoves<CoordsType::POS> move({0, 4}, {0, 1, 2, 3},
      {p0.R[0], p0.R[1], p0.R[2], p0.R[3]});
  ParticleSet::ParticleGradient g(4);
  ParticleSet::ParticleLaplacian l(4);
  RefVector<ParticleSet::ParticleGradient> gradients{g};
  RefVector<ParticleSet::ParticleLaplacian> laplacians{l};
  std::vector<C> ratios(1, C{9, 7});
  ResourceCollection prototype("CPU collective exact cap");
  family.createResource(prototype);
  {
    ResourceCollection resource(prototype);
    const auto cursor = resource.getCursor();
    family.acquireResource(resource, one);
    if (first_fails)
    {
      CHECK_THROWS_AS(family.mw_evaluateMultiParticleMove(one, target, move, ratios, gradients, laplacians), std::length_error);
      CHECK(family.numericStorage().shared_cpu_bytes == before.shared_cpu_bytes);
      CHECK(family.numericStorage().shared_cpu_fingerprint == before.shared_cpu_fingerprint);
      CHECK(ratios[0] == (C{9, 7}));
    }
    else
    {
      family.mw_evaluateMultiParticleMove(one, target, move, ratios, gradients, laplacians);
      CHECK(family.numericStorage().shared_cpu_bytes == cap);
      family.mw_accept_rejectMultiParticleMove(one, target, move, std::vector<bool>{false});
    }
    resource.rewind(cursor);
    family.releaseResource(resource, one);
    if (!first_fails)
    {
      const auto warm = family.numericStorage();
      resource.rewind(cursor);
      RefVectorWithLeader<WaveFunctionComponent> two(family, {family, clone});
      RefVectorWithLeader<ParticleSet> targets(p0, {p0, p1});
      family.acquireResource(resource, two);
      const MCMultiParticleMoves<CoordsType::POS> both({0, 4, 8}, {0, 1, 2, 3, 0, 1, 2, 3},
          {p0.R[0], p0.R[1], p0.R[2], p0.R[3], p1.R[0], p1.R[1], p1.R[2], p1.R[3]});
      ParticleSet::ParticleGradient g1(4);
      ParticleSet::ParticleLaplacian l1(4);
      RefVector<ParticleSet::ParticleGradient> gs{g, g1};
      RefVector<ParticleSet::ParticleLaplacian> ls{l, l1};
      ratios.assign(2, C{9, 7});
      CHECK_THROWS_AS(family.mw_evaluateMultiParticleMove(two, targets, both, ratios, gs, ls), std::length_error);
      CHECK(family.numericStorage().shared_cpu_bytes == warm.shared_cpu_bytes);
      CHECK(family.numericStorage().shared_cpu_fingerprint == warm.shared_cpu_fingerprint);
      CHECK(ratios[0] == (C{9, 7}));
      resource.rewind(cursor);
      family.releaseResource(resource, two);
    }
  }
  CHECK(family.numericStorage().shared_cpu_bytes == before.shared_cpu_bytes);
  CHECK(family.numericStorage().shared_cpu_fingerprint == before.shared_cpu_fingerprint);
}

TEST_CASE("Complex CPU collective crowds evaluate concurrently without shared proposed products",
          "[wavefunction][psiformer][complex_wf][complex_cpu_collective]")
{
  const SimulationCell simulation_cell(cell());
  auto p0 = electrons(simulation_cell), p1 = electrons(simulation_cell);
  auto p2 = electrons(simulation_cell), p3 = electrons(simulation_cell);
  std::array<ParticleSet*, 4> particles{&p0, &p1, &p2, &p3};
  for (std::size_t i = 0; i < particles.size(); ++i)
  {
    particles[i]->R[1][0] += .03 * i;
    particles[i]->update();
  }
  auto source = ions(simulation_cell);
  const std::array<double, 3> twist{.17, -.11, .07};
  PsiFormerComplexWF family("parallel", parameters(), p0, source, clone_cap, twist, 0, {}, "all_electron", clone_cap);
  auto c1 = family.makeClone(p1), c2 = family.makeClone(p2), c3 = family.makeClone(p3);
  auto& leader = dynamic_cast<PsiFormerComplexWF&>(*c2);
  std::array<WaveFunctionComponent*, 4> components{&family, c1.get(), c2.get(), c3.get()};
  std::array<C, 4> initial_logs, expected;
  for (std::size_t i = 0; i < 4; ++i)
  {
    initial_logs[i] = evaluate(*components[i], *particles[i]);
    auto proposed = electrons(simulation_cell);
    proposed.R = particles[i]->R;
    for (std::size_t e = 0; e < 4; ++e)
      proposed.R[e] += ParticleSet::PosType{.001 * (i + 1), -.002, .003};
    proposed.update();
    PsiFormerComplexWF oracle("oracle", parameters(), proposed, source, clone_cap, twist);
    expected[i] = evaluate(oracle, proposed) - initial_logs[i];
  }
  RefVectorWithLeader<WaveFunctionComponent> team0(family, {family, *c1}), team1(leader, {leader, *c3});
  RefVectorWithLeader<ParticleSet> targets0(p0, {p0, p1}), targets1(p2, {p2, p3});
  ResourceCollection prototype("concurrent CPU collective");
  family.createResource(prototype);
  ResourceCollection r0(prototype), r1(prototype);
  const auto cursor0 = r0.getCursor(), cursor1 = r1.getCursor();
  family.acquireResource(r0, team0);
  leader.acquireResource(r1, team1);
  std::promise<void> go;
  const auto start = go.get_future().share();
  std::atomic<int> ready{0};
  auto run = [&](PsiFormerComplexWF& owner, auto& team, auto& targets, std::size_t offset) {
    std::vector<ParticleSet::PosType> positions;
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t e = 0; e < 4; ++e)
        positions.push_back(targets[i].R[e] + ParticleSet::PosType{.001 * (offset + i + 1), -.002, .003});
    const MCMultiParticleMoves<CoordsType::POS> moves({0, 4, 8}, {0, 1, 2, 3, 0, 1, 2, 3}, positions);
    ParticleSet::ParticleGradient g0(4), g1(4);
    ParticleSet::ParticleLaplacian l0(4), l1(4);
    RefVector<ParticleSet::ParticleGradient> gs{g0, g1};
    RefVector<ParticleSet::ParticleLaplacian> ls{l0, l1};
    std::vector<C> ratios(2), result;
    ++ready;
    start.wait();
    for (int repeat = 0; repeat < 5; ++repeat)
    {
      g0 = g1 = C{};
      l0 = l1 = C{};
      owner.mw_evaluateMultiParticleMove(team, targets, moves, ratios, gs, ls);
      result.insert(result.end(), ratios.begin(), ratios.end());
      owner.mw_accept_rejectMultiParticleMove(team, targets, moves, std::vector<bool>{false, false});
    }
    return result;
  };
  auto work0 = std::async(std::launch::async, [&] { return run(family, team0, targets0, 0); });
  auto work1 = std::async(std::launch::async, [&] { return run(leader, team1, targets1, 2); });
  while (ready.load() != 2)
    std::this_thread::yield();
  go.set_value();
  const std::array<std::vector<C>, 2> results{work0.get(), work1.get()};
  for (std::size_t crowd = 0; crowd < 2; ++crowd)
    for (std::size_t i = 0; i < results[crowd].size(); ++i)
      close(results[crowd][i], expected[2 * crowd + i % 2], 2e-12);
  for (std::size_t i = 0; i < 4; ++i)
    CHECK(components[i]->get_log_value() == initial_logs[i]);
  r0.rewind(cursor0);
  r1.rewind(cursor1);
  family.releaseResource(r0, team0);
  leader.releaseResource(r1, team1);
}

TEST_CASE("Complex CPU crowd leases and scalar fallback retain independent products",
          "[wavefunction][psiformer][complex_wf][complex_cpu_shared]")
{
  const SimulationCell simulation_cell(cell());
  auto p0 = electrons(simulation_cell), p1 = electrons(simulation_cell), p2 = electrons(simulation_cell);
  auto p3 = electrons(simulation_cell), p4 = electrons(simulation_cell);
  p1.R[0][0] += .13;
  p2.R[1][2] -= .09;
  p3.R[2][1] += .12;
  p4.R[3][0] -= .07;
  p1.update();
  p2.update();
  p3.update();
  p4.update();
  auto source = ions(simulation_cell);
  const std::array<double, 3> twist{.17, -.11, .07};
  PsiFormerComplexWF family("family", parameters(), p0, source, clone_cap, twist, 0, {}, "all_electron", clone_cap);
  auto clone1 = family.makeClone(p1), clone2 = family.makeClone(p2);
  auto clone3 = family.makeClone(p3), clone4 = family.makeClone(p4);
  auto& crowd1 = dynamic_cast<PsiFormerComplexWF&>(*clone1);
  auto& fallback = dynamic_cast<PsiFormerComplexWF&>(*clone2);
  auto& fallback2 = dynamic_cast<PsiFormerComplexWF&>(*clone3);
  auto& same_crowd = dynamic_cast<PsiFormerComplexWF&>(*clone4);
  PsiFormerComplexWF ref0("ref", parameters(), p0, source, clone_cap, twist);
  PsiFormerComplexWF ref1("ref", parameters(), p1, source, clone_cap, twist);
  PsiFormerComplexWF ref2("ref", parameters(), p2, source, clone_cap, twist);
  PsiFormerComplexWF ref3("ref", parameters(), p3, source, clone_cap, twist);
  PsiFormerComplexWF ref4("ref", parameters(), p4, source, clone_cap, twist);
  const auto expected0 = evaluate(ref0, p0), expected1 = evaluate(ref1, p1), expected2 = evaluate(ref2, p2);
  const auto expected_g0 = p0.G, expected_g1 = p1.G, expected_g2 = p2.G;
  const auto expected3 = evaluate(ref3, p3), expected4 = evaluate(ref4, p4);
  const auto expected_g3 = p3.G, expected_g4 = p4.G;
  RefVectorWithLeader<WaveFunctionComponent> team0(family, {family, same_crowd}), team1(crowd1, {crowd1});
  ResourceCollection prototype("CPU shared");
  family.createResource(prototype);
  ResourceCollection resource0(prototype), resource1(prototype), missing("missing");
  CHECK_THROWS(family.acquireResource(missing, team0));
  RefVectorWithLeader<WaveFunctionComponent> duplicate(family, {family, family});
  CHECK_THROWS(family.acquireResource(resource0, duplicate));
  p0.makeMove(0, ParticleSet::PosType{.01, .02, -.01});
  CHECK_THROWS(family.acquireResource(resource0, team0));
  p0.rejectMove(0);
  const auto cursor0 = resource0.getCursor(), cursor1 = resource1.getCursor();
  family.acquireResource(resource0, team0);
  crowd1.acquireResource(resource1, team1);
  CHECK(family.numericStorage().shared_cpu_owners == 3);
  CHECK(family.numericStorage().shared_cpu_bytes == 0);
  CHECK(family.supportsMultiParticleMoves());
  auto exercise = [](PsiFormerComplexWF& component, ParticleSet& particles) {
    std::vector<C> products;
    for (int repeat = 0; repeat < 12; ++repeat)
    {
      products.push_back(evaluate(component, particles));
      for (int e = 0; e < 4; ++e)
      {
        const auto gradient = component.evalGrad(particles, e);
        for (int d = 0; d < 3; ++d)
          products.push_back(gradient[d]);
      }
    }
    return products;
  };
  std::promise<void> go;
  const auto start = go.get_future().share();
  std::atomic<int> ready{0};
  auto work0 = std::async(std::launch::async, [&] {
    ++ready; start.wait();
    std::vector<C> result;
    for (int repeat = 0; repeat < 4; ++repeat)
    {
      const auto first = exercise(family, p0), second = exercise(same_crowd, p4);
      result.insert(result.end(), first.begin(), first.end());
      result.insert(result.end(), second.begin(), second.end());
    }
    return result;
  });
  auto launch = [&](PsiFormerComplexWF& component, ParticleSet& particles) {
    return std::async(std::launch::async, [&component, &particles, &ready, start, &exercise] {
      ++ready; start.wait(); return exercise(component, particles);
    });
  };
  auto work1 = launch(crowd1, p1), work2 = launch(fallback, p2), work3 = launch(fallback2, p3);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (ready.load() != 4 && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  go.set_value();
  REQUIRE(ready.load() == 4);
  const auto check = [](const std::vector<C>& products, C log, const ParticleSet::ParticleGradient& gradients) {
    for (std::size_t base = 0; base < products.size(); base += 13)
    {
      close(products[base], log, 2e-12);
      for (int e = 0; e < 4; ++e)
        for (int d = 0; d < 3; ++d)
          close(products[base + 1 + 3 * e + d], gradients[e][d], 2e-12);
    }
  };
  const auto team_products = work0.get();
  for (std::size_t base = 0; base < team_products.size(); base += 2 * 12 * 13)
  {
    check(std::vector<C>(team_products.begin() + base, team_products.begin() + base + 12 * 13), expected0, expected_g0);
    check(std::vector<C>(team_products.begin() + base + 12 * 13, team_products.begin() + base + 2 * 12 * 13),
          expected4, expected_g4);
  }
  check(work1.get(), expected1, expected_g1);
  check(work2.get(), expected2, expected_g2);
  check(work3.get(), expected3, expected_g3);
  const auto warm = family.numericStorage();
  CHECK(warm.shared_cpu_bytes == 3 * (warm.shared_cpu_full_bytes + warm.shared_cpu_active_bytes));
  CHECK_THROWS(family.releaseResource(resource0, team0)); // Caller must rewind the loan cursor.
  resource0.rewind(cursor0);
  resource1.rewind(cursor1);
  RefVectorWithLeader<WaveFunctionComponent> wrong_order(family, {same_crowd, family});
  CHECK_THROWS(family.releaseResource(resource0, wrong_order));
  CHECK_THROWS(family.releaseResource(resource1, team0));
  family.releaseResource(resource0, team0);
  crowd1.releaseResource(resource1, team1);
  CHECK(resource0.getCursor() == cursor0 + 1);
  CHECK(resource1.getCursor() == cursor1 + 1);
  CHECK(family.numericStorage().shared_cpu_bytes == warm.shared_cpu_bytes);
  CHECK(family.numericStorage().shared_cpu_fingerprint == warm.shared_cpu_fingerprint);
  CHECK(family.numericStorage().shared_cpu_owners == 3); // Dormant lanes remain charged.
  resource0.rewind(cursor0);
  family.acquireResource(resource0, team0);
  close(evaluate(family, p0), expected0, 2e-12);
  CHECK(family.numericStorage().shared_cpu_bytes == warm.shared_cpu_bytes);
  resource0.rewind(cursor0);
  family.releaseResource(resource0, team0);
  // Destruction, unlike release, returns dormant numerical reservations.
  {
    ResourceCollection temporary(prototype);
    RefVectorWithLeader<WaveFunctionComponent> temporary_team(fallback2, {fallback2});
    fallback2.acquireResource(temporary, temporary_team);
    close(evaluate(fallback2, p3), expected3, 2e-12);
    CHECK(family.numericStorage().shared_cpu_owners == 4);
    CHECK(family.numericStorage().shared_cpu_bytes == warm.shared_cpu_bytes + warm.shared_cpu_full_bytes);
    temporary.rewind();
    fallback2.releaseResource(temporary, temporary_team);
  }
  CHECK(family.numericStorage().shared_cpu_owners == 3);
  CHECK(family.numericStorage().shared_cpu_bytes == warm.shared_cpu_bytes);
  CHECK(family.numericStorage().shared_cpu_fingerprint == warm.shared_cpu_fingerprint);
}

TEST_CASE("PsiFormer complex walker record roundtrips only matching FULL own products",
          "[wavefunction][psiformer][complex_wf]")
{
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell);
  auto branch_p = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  PsiFormerComplexWF component("buffer", parameters(), p, source, clone_cap, {.17, -.11, .07});
  const auto storage = component.numericStorage();
  WaveFunctionComponent::WFBufferType buffer;
  // Prefixes exercise the independent bulk/scalar cursors used by the real walker contract.
  std::array<double, 2> bulk_prefix{};
  double scalar_prefix = 0;
  buffer.add(bulk_prefix.data(), bulk_prefix.data() + bulk_prefix.size());
  buffer.add(scalar_prefix);
  const auto bulk_start = buffer.current(), scalar_start = buffer.current_scalar();
  component.registerData(p, buffer);
  const auto bulk_end = buffer.current(), scalar_end = buffer.current_scalar();
  CHECK(bulk_end == bulk_start);
  CHECK(scalar_end - scalar_start == 29 + 8 * 4);
  buffer.allocate();
  buffer.rewind(bulk_start, scalar_start);
  component.copyFromBuffer(p, buffer); // freshly allocated zero sentinel is invalid, not accepted
  CHECK(buffer.current_scalar() == scalar_end);
  CHECK_THROWS(component.observeCoulombEnergy(p, -2.7));
  const C gradient_sentinel{.12, -.04}, laplacian_sentinel{-.31, .07};
  p.G = gradient_sentinel;
  p.L = laplacian_sentinel;
  buffer.rewind(bulk_start, scalar_start);
  const auto initial_log = component.updateBuffer(p, buffer, true);
  const auto initial_g = p.G;
  const auto initial_l = p.L;
  const auto energy = component.observeCoulombEnergy(p, -2.7);
  auto branch_base = component.makeClone(branch_p);
  auto& branch = dynamic_cast<PsiFormerComplexWF&>(*branch_base);
  const auto branch_storage = branch.numericStorage();
  branch_p.G = gradient_sentinel;
  branch_p.L = laplacian_sentinel;
  buffer.rewind(bulk_start, scalar_start);
  branch.copyFromBuffer(branch_p, buffer);
  for (int e = 0; e < 4; ++e)
  {
    close(branch_p.L[e], laplacian_sentinel); // restoration does not scatter caller totals
    for (int d = 0; d < 3; ++d)
      close(branch_p.G[e][d], gradient_sentinel);
  }
  close(branch.observeCoulombEnergy(branch_p, -2.7).local_energy, energy.local_energy);
  buffer.rewind(bulk_start, scalar_start);
  close(branch.updateBuffer(branch_p, buffer), initial_log);
  for (int e = 0; e < 4; ++e)
  {
    close(branch_p.L[e], initial_l[e]);
    for (int d = 0; d < 3; ++d)
      close(branch_p.G[e][d], initial_g[e][d]);
  }

  // Corruption, foreign model/twist and moved configurations never publish accepted caches.
  for (auto offset : {std::size_t{0}, std::size_t{18}, scalar_end - scalar_start - 3})
  {
    auto corrupt = buffer;
    corrupt.Scalar_ptr[scalar_start + offset] += .125;
    corrupt.rewind(bulk_start, scalar_start);
    CHECK_THROWS(branch.copyFromBuffer(branch_p, corrupt));
    CHECK_THROWS(branch.observeCoulombEnergy(branch_p, -2.7));
    buffer.rewind(bulk_start, scalar_start);
    branch.copyFromBuffer(branch_p, buffer);
  }
  auto foreign_parameters = parameters();
  foreign_parameters.values.front() += .001;
  PsiFormerComplexWF foreign("foreign", std::move(foreign_parameters), branch_p, source,
                             clone_cap, {.17, -.11, .07});
  buffer.rewind(bulk_start, scalar_start);
  CHECK_THROWS(foreign.copyFromBuffer(branch_p, buffer));
  PsiFormerComplexWF wrong_twist("twist", parameters(), branch_p, source, clone_cap);
  buffer.rewind(bulk_start, scalar_start);
  CHECK_THROWS(wrong_twist.copyFromBuffer(branch_p, buffer));
  const auto saved_position = branch_p.R[0];
  branch_p.R[0][0] += .125;
  branch_p.update();
  buffer.rewind(bulk_start, scalar_start);
  CHECK_THROWS(branch.copyFromBuffer(branch_p, buffer));
  branch_p.R[0] = saved_position;
  branch_p.update();
  WaveFunctionComponent::WFBufferType insufficient;
  insufficient.add(scalar_prefix);
  insufficient.allocate();
  insufficient.rewind();
  CHECK_THROWS(branch.copyFromBuffer(branch_p, insufficient));
  buffer.rewind(bulk_start, scalar_start);
  branch.copyFromBuffer(branch_p, buffer);

  // VALUE and ACTIVE accepts retain scalar validity, but require a new FULL observation/record.
  for (bool active : {false, true})
  {
    branch_p.makeMove(2, ParticleSet::PosType{.007, -.011, .013});
    WaveFunctionComponent::GradType gradient{};
    if (active)
      branch.ratioGrad(branch_p, 2, gradient);
    else
      branch.ratio(branch_p, 2);
    CHECK_THROWS(branch.observeCoulombEnergy(branch_p, -2.7));
    branch.acceptMove(branch_p, 2);
    branch_p.acceptMove(2);
    CHECK_THROWS(branch.observeCoulombEnergy(branch_p, -2.7));
    branch_p.G = 0;
    branch_p.L = 0;
    buffer.rewind(bulk_start, scalar_start);
    branch.updateBuffer(branch_p, buffer);
    CHECK_NOTHROW(branch.observeCoulombEnergy(branch_p, -2.7));
  }
  branch_p.makeMove(0, ParticleSet::PosType{.01, .02, -.01});
  branch.ratio(branch_p, 0);
  buffer.rewind(bulk_start, scalar_start);
  CHECK_THROWS(branch.updateBuffer(branch_p, buffer));
  CHECK_THROWS(branch.acceptMove(branch_p, 0));
  branch_p.rejectMove(0);
  evaluate(branch, branch_p);
  CHECK(branch.numericStorage().fingerprint == branch_storage.fingerprint);
  CHECK(branch.numericStorage().allocated_bytes == branch_storage.allocated_bytes);
  CHECK(component.numericStorage().fingerprint == storage.fingerprint);
  CHECK(component.numericStorage().allocated_bytes == storage.allocated_bytes);
}

TEST_CASE("PsiFormer complex Coulomb observation retains mass-weighted imaginary energy",
          "[wavefunction][psiformer][complex_wf]")
{
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell);
  auto gamma_p = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  for (auto* target : {&p, &gamma_p})
  {
    const int mass = target->getSpeciesSet().addAttribute("mass");
    target->getSpeciesSet()(mass, 0) = .7;
    target->getSpeciesSet()(mass, 1) = 2.3;
    target->resetGroups();
  }
  PsiFormerComplexWF component("energy", parameters(), p, source, clone_cap, {.17, -.11, .07});
  PsiFormerComplexWF gamma("gamma", parameters(), gamma_p, source, clone_cap);
  evaluate(component, p);
  evaluate(gamma, gamma_p);
  // Synthetic same-sample V isolates this helper; periodic Coulomb production is gated at XML integration.
  constexpr double potential = -2.7;
  const auto observation = component.observeCoulombEnergy(p, potential);
  C independent{}, twist_change{};
  for (int e = 0; e < 4; ++e)
  {
    const double inverse_mass = 1.0 / (e < 2 ? .7 : 2.3);
    C ratio = p.L[e];
    for (int d = 0; d < 3; ++d)
    {
      ratio += p.G[e][d] * p.G[e][d];
      const C delta = p.G[e][d] - gamma_p.G[e][d];
      close(C{delta.real()}, C{});
      const double k = delta.imag();
      twist_change += inverse_mass * (.5 * k * k - C{0, k} * gamma_p.G[e][d]);
    }
    independent -= .5 * inverse_mass * ratio;
  }
  close(observation.kinetic, independent);
  close(observation.local_energy, independent + potential);
  close(observation.kinetic - gamma.observeCoulombEnergy(gamma_p, potential).kinetic, twist_change);
  CHECK(std::abs(observation.kinetic.imag()) > 1e-6);
  RuntimeOptions options;
  TrialWaveFunction unused(options);
  BareKineticEnergy production_kinetic(p);
  CHECK(observation.kinetic.real() == Approx(production_kinetic.evaluate(unused, p)).epsilon(2e-12));
  p.G = C{100, 200};
  p.L = C{300, -400};
  close(component.observeCoulombEnergy(p, potential).local_energy, observation.local_energy);
  CHECK_THROWS(component.observeCoulombEnergy(p, std::numeric_limits<double>::quiet_NaN()));
  const int mass = p.getSpeciesSet().addAttribute("mass");
  p.getSpeciesSet()(mass, 1) = 0;
  p.resetGroups();
  CHECK_THROWS(component.observeCoulombEnergy(p, potential));
}

TEST_CASE("PsiFormer paired XML rejects unsafe requests before target or device mutation",
          "[wavefunction][psiformer][complex_wf][complex_admission]")
{
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell);
  PsiFormerWaveFunctionBuilder::PSetMap pool;
  pool.emplace("ion0", std::make_unique<ParticleSet>(ions(simulation_cell)));
  AdmissionReadiness readiness;
  PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, p, pool, &readiness);
  auto build = [&](const std::string& xml) {
    Libxml2Document document;
    REQUIRE(document.parseFromString(xml));
    return builder.buildComponent(document.getRoot());
  };
  for (const char* request : {"gpu='auto'", "gpu='cuda'", "gpu='cuda' max_gpu_numeric_bytes='0'",
                             "max_gpu_numeric_bytes='1024'", "optimize='yes'",
                             "precision='fp32_dense_fp64_islands'"})
    CHECK_THROWS(build(complexXml(request)));
  CHECK_THROWS(build(complexXml("", "")));
  CHECK_THROWS(build(complexXml("", "0")));
  CHECK_THROWS(build(complexXml("parameter_checkpoint=''")));
  CHECK_THROWS(build("<psiformer initialization='deepqmc_psiformer_v1' parameter_checkpoint='unused.root.h5'/>"));
  CHECK_THROWS(build(complexXml("twist='0.17 -0.11 0.07'", "1")));
  CHECK(readiness.calls == 0);
  // AE HIP is an admitted profile, but this explicit rejecting readiness seam remains closed.
  CHECK_THROWS_WITH(build(complexXml("gpu='hip' max_gpu_numeric_bytes='67108864' "
                                    "optimize='no' optimize_scope='indices'")),
                    Catch::Matchers::ContainsSubstring("PsiFormer accelerator backend 'hip' is not production-ready"));
  CHECK(readiness.calls == 1);
  for (int d = 0; d < 3; ++d)
    CHECK(p.getTwist()[d] == 0);
  auto& source = *pool.at("ion0");
  const int valence = source.getSpeciesSet().findAttribute("valence");
  source.getSpeciesSet()(valence, 0) = 2;
  CHECK_THROWS(build(complexXml())); // ECP effective charges cannot masquerade as AE metadata
}

TEST_CASE("Paired CPU XML admits complete structured optimization and preserves narrow guards",
          "[wavefunction][psiformer][complex_wf][complex_admission]")
{
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell);
  PsiFormerWaveFunctionBuilder::PSetMap pool;
  pool.emplace("ion0", std::make_unique<ParticleSet>(ions(simulation_cell)));
  AdmissionReadiness readiness;
  PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, p, pool, &readiness);
  const auto build = [&](const std::string& xml) {
    Libxml2Document document;
    REQUIRE(document.parseFromString(xml));
    return builder.buildComponent(document.getRoot());
  };
  auto component = build(complexXml("gpu='cpu' optimize='yes' optimize_scope='all'"));
  auto* provider = component->structuredParameterProvider();
  REQUIRE(provider != nullptr);
  REQUIRE_FALSE(provider->affineParameterBlocks().empty());
  CHECK(std::all_of(provider->parameterSchema().blocks().begin(), provider->parameterSchema().blocks().end(),
                    [](const auto& block) { return block.trainable; }));
  CHECK_THROWS(build(complexXml("gpu='cpu' optimize='yes' optimize_indices='0'")));
  CHECK_THROWS(build(complexXml("gpu='cuda' optimize='yes' optimize_scope='all'")));
  CHECK(readiness.calls == 0);
}

TEST_CASE("Legacy paired CPU pseudopotential inference initializes without optimization",
          "[wavefunction][psiformer][complex_wf][complex_cpu_pp_inference]")
{
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell);
  p.create({1, 1});
  p.R[0] = {.2, .1, .3}; p.R[1] = {1.1, .5, 1.4};
  p.resetGroups(); p.update();
  auto training_p = ParticleSet(p);
  PsiFormerWaveFunctionBuilder::PSetMap pool;
  auto source = std::make_unique<ParticleSet>(ions(simulation_cell));
  auto& species = source->getSpeciesSet();
  species(species.findAttribute("charge"), 0) = 1;
  species(species.findAttribute("valence"), 0) = 1;
  source->resetGroups(); source->update();
  pool.emplace("ion0", std::move(source));
  AdmissionReadiness readiness;
  PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, p, pool, &readiness);
  PsiFormerWaveFunctionBuilder training_builder(OHMMS::Controller, training_p, pool, &readiness);
  const auto xml = [](const std::string& controls) {
    return std::string("<psiformer source='ion0' system='pseudopotential' ") +
        "initialization='psiformer_complex_periodic_v1' initialization_seed='17' " +
        "feature_policy='periodic_torus_v1' gpu='cpu' max_clone_numeric_bytes='67108864' " + controls + "/>";
  };
  const auto build = [](PsiFormerWaveFunctionBuilder& selected, const std::string& text) {
    Libxml2Document document;
    REQUIRE(document.parseFromString(text));
    return selected.buildComponent(document.getRoot());
  };
  for (const std::string twist : {"0 0 0", ".125 -.0625 .03125"})
  {
    p.setTwist({}); training_p.setTwist({});
    const auto fixed_twist = "twist='" + twist + "' ";
    auto cold_base = build(builder, xml(fixed_twist + "optimize='no'"));
    auto training_base = build(training_builder, xml(fixed_twist + "optimize='yes' optimize_scope='all'"));
    auto* cold = dynamic_cast<PsiFormerComplexWF*>(cold_base.get());
    auto* training = dynamic_cast<PsiFormerComplexWF*>(training_base.get());
    REQUIRE(cold != nullptr); REQUIRE(training != nullptr);
    CHECK_FALSE(cold->gpuInferenceEnabled());
    CHECK(cold->isPseudopotentialSystem());
    CHECK(cold->snapshotParameters().version == 0);
    CHECK(cold->snapshotParameters().values == training->snapshotParameters().values);
    close(evaluate(*cold, p), evaluate(*training, training_p), 2e-12);
    for (int e = 0; e < p.getTotalNum(); ++e)
    {
      close(p.L[e], training_p.L[e], 2e-12);
      for (int d = 0; d < 3; ++d) close(p.G[e][d], training_p.G[e][d], 2e-12);
    }
    close(cold->observeCoulombEnergy(p, .25).local_energy,
          training->observeCoulombEnergy(training_p, .25).local_energy, 2e-12);
  }
  CHECK_THROWS(build(builder, xml("optimize='no' optimize_scope='all'")));
  CHECK_THROWS(build(builder, xml("optimize='no' precision='fp32_dense_fp64_islands'")));
  CHECK(readiness.calls == 0);
}

TEST_CASE("PsiFormer paired XML observes the same production Coulomb Hamiltonian sample",
          "[wavefunction][psiformer][complex_wf][complex_admission]")
{
  auto lattice = cell();
  lattice.LR_dim_cutoff = 15;
  lattice.reset();
  const SimulationCell simulation_cell(lattice);
  auto p = electrons(simulation_cell);
  const int charge = p.getSpeciesSet().addAttribute("charge");
  p.getSpeciesSet()(charge, 0) = p.getSpeciesSet()(charge, 1) = -1;
  p.resetGroups();
  PsiFormerWaveFunctionBuilder::PSetMap pool;
  pool.emplace("ion0", std::make_unique<ParticleSet>(ions(simulation_cell)));
  auto& source = *pool.at("ion0");
  source.createSK();
  p.createSK();
  PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, p, pool);
  Libxml2Document document;
  REQUIRE(document.parseFromString(complexXml("twist='0.17 -0.11 0.07'")));
  auto base = builder.buildComponent(document.getRoot());
  auto* component = dynamic_cast<PsiFormerComplexWF*>(base.get());
  REQUIRE(component != nullptr);
  CHECK(&component->sourceParticles() == &source);
  CHECK(component->numericStorage().required_bytes == component->numericStorage().allocated_bytes);
  RuntimeOptions options;
  TrialWaveFunction trial(options);
  trial.addComponent(std::move(base));
  auto makeHamiltonian = [&](const std::string& xml) {
    Libxml2Document input;
    REQUIRE(input.parseFromString(xml));
    HamiltonianFactory factory("complex_coulomb", p, pool, OptionalRef<TrialWaveFunction>(trial), OHMMS::Controller);
    if (!factory.put(input.getRoot()))
      throw std::runtime_error("Hamiltonian construction returned false");
    return factory.releaseHamiltonian();
  };
  CHECK_THROWS(makeHamiltonian("<hamiltonian default='no'>" + coulombTerms + "</hamiltonian>"));
  CHECK_THROWS(makeHamiltonian("<hamiltonian><pairpot type='pseudo' source='ion0'/></hamiltonian>"));
  CHECK_THROWS(makeHamiltonian("<hamiltonian><pairpot type='coulomb' source='other'/></hamiltonian>"));
  auto hamiltonian = makeHamiltonian("<hamiltonian>" + coulombTerms + "</hamiltonian>");
  source.update();
  p.update();
  // One exact configuration is used for BOTH the production Ewald/Coulomb traversal and helper.
  trial.evaluateLog(p);
  const auto real_energy = hamiltonian->evaluate(trial, p);
  const auto potential = hamiltonian->getLocalPotential();
  const auto complete = component->observeCoulombEnergy(p, potential);
  CHECK(complete.local_energy.real() == Approx(real_energy).epsilon(2e-11));
  close(complete.local_energy - complete.kinetic, C{potential});
  C independent{};
  for (int e = 0; e < 4; ++e)
  {
    C ratio = p.L[e];
    for (int d = 0; d < 3; ++d)
      ratio += p.G[e][d] * p.G[e][d];
    independent -= .5 * ratio;
  }
  close(complete.local_energy, independent + potential);
  CHECK(std::abs(complete.local_energy.imag()) > 1e-8);
  trial.addComponent(component->makeClone(p));
  CHECK_THROWS(makeHamiltonian("<hamiltonian>" + coulombTerms + "</hamiltonian>"));
}

TEST_CASE("PsiFormer public inference rejects unsafe inherited fallback APIs",
          "[wavefunction][psiformer][complex_wf][complex_admission]")
{
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  PsiFormerComplexWF component("reject", parameters(), p, source, clone_cap);
  WaveFunctionComponent::ComplexType spin{.2, -.1};
  WaveFunctionComponent::GradType gradient(C{.3, .4});
  WaveFunctionComponent::HessVector hessian;
  std::vector<WaveFunctionComponent::ValueType> ratios(1, C{.7, -.2});
  VirtualParticleSet virtuals(p);
  CHECK_THROWS(component.evalGradSource(p, source, 0));
  CHECK_THROWS(component.evaluateHessian(p, hessian));
  CHECK_THROWS(component.evalGradWithSpin(p, 0, spin));
  CHECK_THROWS(component.ratioGradWithSpin(p, 0, gradient, spin));
  CHECK_THROWS(component.evaluateRatiosAlltoOne(p, ratios));
  CHECK_THROWS(component.evaluateRatios(virtuals, ratios));
  close(spin, C{.2, -.1});
  close(gradient[0], C{.3, .4});
  close(ratios[0], C{.7, -.2});
}

TEST_CASE("Real Gamma pair components retain complete physical ions and profile persistence",
          "[wavefunction][psiformer][complex_wf][real_periodic_profile]")
{
  for (bool two_cells : {false, true})
  {
    const SimulationCell simulation_cell(cell());
    auto p = electrons(simulation_cell), clone_p = electrons(simulation_cell);
    ParticleSet source(simulation_cell);
    source.setName("ion0");
    source.create(two_cells ? std::vector<int>{2, 2} : std::vector<int>{1, 1});
    auto& species = source.getSpeciesSet();
    species.addSpecies("Li"); species.addSpecies("H");
    const int charge = species.addAttribute("charge");
    species(charge, 0) = 3; species(charge, 1) = 1;
    source.resetGroups();
    source.R[0] = {0, 0, 0};
    if (two_cells)
    {
      source.R[1] = {4, 0, 0}; source.R[2] = {1, 1, 1}; source.R[3] = {5, 1, 1};
    }
    else source.R[1] = {1, 1, 1};
    source.update();
    auto initialized = parameters(false);
    initialized.profile = ps::PSIFORMER_PERIODIC_PAIR_V1;
    auto& environment = initialized.environment;
    environment.boundary = ps::BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = ps::GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1;
    environment.periodic_axes = {true, true, true};
    for (int a = 0; a < 3; ++a)
      for (int d = 0; d < 3; ++d) environment.lattice_vectors[a][d] = cell().R(a, d);
    auto metadata = std::make_shared<ps::PeriodicProfileMetadata>();
    metadata->cells.primitive = metadata->cells.simulation =
        {pf::GeometryBoundaryKind::PERIODIC, environment.lattice_vectors, environment.periodic_axes};
    if (two_cells)
    {
      metadata->cells.primitive.lattice_vectors[0][0] = 4;
      metadata->cells.tiling[0][0] = 2;
    }
    metadata->primitive_motif = {{0, 0, 0}, {1, 1, 1}};
    metadata->motif_species = {0, 1};
    metadata->feature_transform = ps::PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1;
    metadata->electron_electron_cusp = false;
    for (std::size_t column = 0; column < 4; ++column)
      metadata->occupied_columns.push_back({{}, column % 2, int(column / 2), 1});
    environment.periodic_profile = metadata;
    for (std::size_t block = 0; block < initialized.model_shape.attention_blocks; ++block)
      for (bool bias : {false, true})
      {
        const auto begin = initialized.values.size(), heads = initialized.model_shape.attention_heads;
        initialized.values.resize(begin + (bias ? 1 : 8) * heads, .02);
        const auto layer = block ? "electron_gnn_layer_" + std::to_string(block) : std::string("electron_gnn_layer");
        initialized.layouts.push_back({"periodic/" + layer + "/pair_attention", bias ? "b" : "w",
            bias ? std::vector<std::size_t>{heads} : std::vector<std::size_t>{8, heads}, begin, initialized.values.size()});
      }
    PsiFormerWF component("real_pair", initialized, p, source, true, {}, true);
    component.validateSystem(p, source, "all_electron");
    CHECK_FALSE(component.matchesRealDevicePairTarget(p)); // Raw/no-cusp family is not the selected device contract.
    auto selected = initialized;
    auto selected_metadata = std::make_shared<ps::PeriodicProfileMetadata>(*metadata);
    selected_metadata->feature_transform = ps::PeriodicFeatureTransform::LOG_COMPLEMENT_V1;
    selected_metadata->electron_electron_cusp = true;
    selected.environment.periodic_profile = selected_metadata;
    PsiFormerWF selected_component("real_pair", selected, p, source, true, {}, true);
    selected_component.validateSystem(p, source, "all_electron");
    CHECK(selected_component.matchesRealDevicePairTarget(p));
    CHECK_FALSE(selected_component.usesAcceleratorParameterProducts());
    const auto training_identity = selected_component.trainingConfigurationFingerprint();
    CHECK(selected_component.parameterSchema().fingerprint() == component.parameterSchema().fingerprint());
    CHECK(training_identity != component.trainingConfigurationFingerprint());
    auto changed_parameters = selected_component.snapshotParameters();
    changed_parameters.values[0] += .001;
    selected_component.publishParameters(changed_parameters, changed_parameters.version);
    CHECK(selected_component.trainingConfigurationFingerprint() == training_identity);
    p.setTwist(ParticleSet::PosType{.125, 0, 0});
    CHECK_FALSE(selected_component.matchesRealDevicePairTarget(p));
    p.setTwist(ParticleSet::PosType{});
    auto mismatched_cell = cell();
    mismatched_cell.R(0, 0) += .25;
    mismatched_cell.reset();
    const SimulationCell different_cell(mismatched_cell);
    auto different_target = electrons(different_cell);
    CHECK_FALSE(selected_component.matchesRealDevicePairTarget(different_target));
    const auto center = evaluate(component, p);
    const auto gradient = p.G;
    const auto laplacian = p.L;
    constexpr double step = 1e-4;
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
      for (int d = 0; d < 3; ++d)
      {
        const auto position = p.R[electron][d];
        p.R[electron][d] = position + step;
        const auto plus = evaluate(component, p);
        p.R[electron][d] = position - step;
        const auto minus = evaluate(component, p);
        p.R[electron][d] = position;
        close((plus - minus) / (2 * step), gradient[electron][d], 2e-6);
      }
    auto clone = component.makeClone(clone_p);
    auto* typed = dynamic_cast<PsiFormerWF*>(clone.get());
    REQUIRE(typed);
    typed->isolateParameterState();
    close(evaluate(*typed, clone_p), center);
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
    {
      close(clone_p.L[electron], laplacian[electron]);
      for (int d = 0; d < 3; ++d) close(clone_p.G[electron][d], gradient[electron][d]);
    }
    const auto replica = two_cells ? 1 : 0;
    source.R[replica][0] += .01;
    CHECK_THROWS(evaluate(component, p)); CHECK_THROWS(evaluate(*typed, clone_p));
    source.R[replica][0] -= .01;
    close(evaluate(component, p), center);
    const auto path = "psiformer_real_pair." + std::to_string(OHMMS::Controller->rank()) + ".h5";
    REQUIRE_FALSE(std::filesystem::exists(path));
    component.exportModel(path);
    auto imported = ps::readPsiFormerModelFile(path);
    CHECK_FALSE(imported.paired_complex);
    REQUIRE(imported.parameters.environment.periodic_profile);
    CHECK(imported.parameters.model_shape.nuclei == 2);
    CHECK(imported.parameters.environment.periodic_profile->cells.tiling[0][0] == (two_cells ? 2 : 1));
    ps::validatePsiFormerModelFile(imported, path, clone_p, source, "all_electron", {});
    source.R[0][0] += 8;
    CHECK_THROWS(ps::validatePsiFormerModelFile(imported, path, clone_p, source, "all_electron", {}));
    source.R[0][0] -= 8;
    PsiFormerWF loaded("real_pair", imported.parameters, clone_p, source, true, {}, true);
    loaded.validateSystem(clone_p, source, "all_electron");
    close(evaluate(loaded, clone_p), center);
    std::size_t receipt = 0;
    const auto fingerprint = [&](PsiFormerWF& model) {
      const auto vp_path = path + ".vp" + std::to_string(receipt++);
      REQUIRE_FALSE(std::filesystem::exists(vp_path));
      {
        hdf_archive output;
        REQUIRE(output.create(vp_path));
        model.writeVariationalParameters(output);
        output.flush();
      }
      std::string result;
      {
        hdf_archive input;
        REQUIRE(input.open(vp_path, H5F_ACC_RDONLY));
        input.push("PsiFormer", false); input.push("real_pair", false);
        input.read(result, "model_fingerprint");
      }
      CHECK(std::filesystem::remove(vp_path));
      return result;
    };
    const auto original_identity = fingerprint(component);
    CHECK(fingerprint(*typed) == original_identity);
    CHECK(fingerprint(loaded) == original_identity);
    CHECK(std::filesystem::remove(path));
  }
}

TEST_CASE("Legacy standalone periodic import accepts only equivalent ordered ionic images",
          "[wavefunction][psiformer][complex_wf][legacy_import_images]")
{
  for (bool paired : {false, true})
  {
    const SimulationCell simulation_cell(cell());
    auto p = electrons(simulation_cell), raw = ions(simulation_cell);
    const std::array<double, 3> twist = paired ? std::array<double, 3>{.17, -.11, .07} : std::array<double, 3>{};
    char pattern[] = "/tmp/qmcpack_import_images_XXXXXX";
    fixture::GeneratedFiles files;
    const auto* directory = mkdtemp(pattern);
    REQUIRE(directory);
    files.directory = directory;
    const auto path = (files.directory / "model.h5").string();

    // Actual XML parsing wraps the exact exported Cartesian coordinates in the skew cell.
    std::ostringstream text;
    text << std::setprecision(17) << "<particleset name='ion0' size='2'>"
         << "<group name='Li'><parameter name='charge'>3</parameter></group>"
         << "<group name='H'><parameter name='charge'>1</parameter></group>"
         << "<attrib name='ionid' datatype='stringArray'>Li H</attrib>"
         << "<attrib name='position' datatype='posArray'>";
    for (const auto& position : raw.R)
      for (int d = 0; d < 3; ++d) text << position[d] << ' ';
    text << "</attrib></particleset>";
    Libxml2Document xml;
    REQUIRE(xml.parseFromString(text.str()));
    ParticleSet wrapped(simulation_cell);
    XMLParticleParser parser(wrapped);
    REQUIRE(parser.readXML(xml.getRoot()));
    CHECK(raw.R[0] != wrapped.R[0]);
    const auto make = [&](const ps::InitializedPsiFormerParameters& initial, ParticleSet& source)
        -> std::unique_ptr<WaveFunctionComponent> {
      if (paired)
        return std::make_unique<PsiFormerComplexWF>("image_model", initial, p, source, clone_cap, twist);
      auto real = std::make_unique<PsiFormerWF>("image_model", initial, p, source, true, std::vector<std::size_t>{},
                                               true, "", twist);
      real->validateSystem(p, source, "all_electron");
      return real;
    };
    auto original = make(parameters(paired), raw);
    if (paired) dynamic_cast<PsiFormerComplexWF&>(*original).exportModel(path);
    else dynamic_cast<PsiFormerWF&>(*original).exportModel(path);
    const auto imported = ps::readPsiFormerModelFile(path);
    REQUIRE_NOTHROW(ps::validatePsiFormerModelFile(imported, path, p, wrapped, "all_electron", twist));
    auto loaded = make(imported.parameters, wrapped);
    CHECK(original->structuredParameterProvider()->snapshotParameters().values ==
          loaded->structuredParameterProvider()->snapshotParameters().values);

    // Ionic image changes do not create an ionic Bloch phase: only electron coordinates own that phase.
    const auto raw_log = evaluate(*original, p);
    const auto raw_g = p.G;
    const auto raw_l = p.L;
    const auto wrapped_log = evaluate(*loaded, p);
    CHECK(raw_log.real() == Approx(wrapped_log.real()).margin(2e-10));
    close(std::polar(1.0, raw_log.imag() - wrapped_log.imag()), C{1}, 2e-10);
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
    {
      close(raw_l[electron], p.L[electron], 2e-10);
      for (int d = 0; d < 3; ++d) close(raw_g[electron][d], p.G[electron][d], 2e-10);
    }
    // Compare full canonical log-score channels, discarding transient affine rows.
    class DiscardRows final : public wftrain::AffineScoreObserver
    {
      void observe(const wftrain::AffineScoreObservation&) override {}
    } discard;
    const auto scores = [&](WaveFunctionComponent& model) {
      std::vector<double> real, imaginary;
      if (paired)
      {
        auto& complex = dynamic_cast<PsiFormerComplexWF&>(model);
        complex.prepareAffineScores(complex.affineScoreStorage().required_bytes);
        complex.observeComplexAffineScores(p, discard, real, imaginary);
      }
      else
      {
        dynamic_cast<PsiFormerWF&>(model).observeAffineScores(p, discard, real);
        imaginary.resize(real.size());
      }
      std::vector<C> result(real.size());
      for (std::size_t i = 0; i < real.size(); ++i) result[i] = {real[i], imaginary[i]};
      return result;
    };
    const auto raw_scores = scores(*original), wrapped_scores = scores(*loaded);
    REQUIRE(raw_scores.size() == wrapped_scores.size());
    for (std::size_t i = 0; i < raw_scores.size(); ++i) close(raw_scores[i], wrapped_scores[i], 2e-10);

    // Geometry, order, charges, cell and twist remain physical metadata, not loose matching hints.
    const double saved = wrapped.R[0][0];
    wrapped.R[0][0] += 1e-4;
    CHECK_THROWS(ps::validatePsiFormerModelFile(imported, path, p, wrapped, "all_electron", twist));
    wrapped.R[0][0] = saved;
    std::swap(wrapped.R[0], wrapped.R[1]);
    CHECK_THROWS(ps::validatePsiFormerModelFile(imported, path, p, wrapped, "all_electron", twist));
    std::swap(wrapped.R[0], wrapped.R[1]);
    const int charge = wrapped.getSpeciesSet().findAttribute("charge");
    wrapped.getSpeciesSet()(charge, 0) = 2;
    CHECK_THROWS(ps::validatePsiFormerModelFile(imported, path, p, wrapped, "all_electron", twist));
    wrapped.getSpeciesSet()(charge, 0) = 3;
    auto wrong_twist = twist;
    wrong_twist[0] += .01;
    CHECK_THROWS(ps::validatePsiFormerModelFile(imported, path, p, wrapped, "all_electron", wrong_twist));
    for (double bad : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
    {
      wrapped.R[0][0] = bad;
      CHECK_THROWS(ps::validatePsiFormerModelFile(imported, path, p, wrapped, "all_electron", twist));
    }
    wrapped.R[0][0] = saved;
    auto changed_cell = cell();
    changed_cell.R(0, 0) += .1;
    changed_cell.reset();
    const SimulationCell other_cell(changed_cell);
    ParticleSet wrong_cell(other_cell);
    XMLParticleParser other_parser(wrong_cell);
    REQUIRE(other_parser.readXML(xml.getRoot()));
    CHECK_THROWS(ps::validatePsiFormerModelFile(imported, path, p, wrong_cell, "all_electron", twist));

    if (!paired)
    {
      // OPEN import keeps exact geometry equality, including a sub-tolerance displacement.
      Lattice open_cell;
      open_cell.R = {1, 0, 0, 0, 1, 0, 0, 0, 1};
      open_cell.BoxBConds = {false, false, false};
      open_cell.reset();
      const SimulationCell open_simulation(open_cell);
      auto open_p = electrons(open_simulation), open_ions = ions(open_simulation);
      PsiFormerWF open("open_images", parameters(false, false, 4), open_p, open_ions, true, {}, true);
      open.validateSystem(open_p, open_ions, "all_electron");
      const auto open_path = (files.directory / "open.h5").string();
      open.exportModel(open_path);
      const auto open_import = ps::readPsiFormerModelFile(open_path);
      REQUIRE_NOTHROW(ps::validatePsiFormerModelFile(open_import, open_path, open_p, open_ions, "all_electron", {}));
      open_ions.R[0][0] += 1e-15;
      CHECK_THROWS(ps::validatePsiFormerModelFile(open_import, open_path, open_p, open_ions, "all_electron", {}));
    }
  }
}

TEST_CASE("Shared DEVICE affine sink rejects an unimplemented imaginary fallback channel",
          "[wavefunction][psiformer][complex_wf][complex_device_affine_sink]")
{
  struct RealSink final : wftrain::DeviceAffineStatisticsSink
  {
    void beginSample() override {}
    void addFallbackScore(const std::vector<double>&) override { ++calls; }
    void endSample() override {}
    void abortSample() noexcept override {}
    void import(std::vector<wftrain::KFACFactorStatistics>) override {}
    int calls = 0;
  } sink;
  CHECK_THROWS(sink.addComplexFallbackScore({1}, {0}));
  CHECK(sink.calls == 0);
  CHECK_NOTHROW(sink.addComplexFallbackScore({1}, {}));
  CHECK(sink.calls == 1);
}

#ifdef ENABLE_CUDA
TEST_CASE("Paired Ne100 complete score and affine factors preserve ordered shared row means",
          "[wavefunction][psiformer][complex_wf][complex_shared_row_means]")
{
  using namespace wftrain;
  namespace dev = ps::device;
  const char* input = std::getenv("PSIFORMER_TEST_PAIRED_SHARED_MEANS_INPUT");
  if (!input) { WARN("NOT_RUN: frozen Ne100 paired input required"); return; }
  int devices = 0;
#ifdef QMC_CUDA2HIP
  REQUIRE(hipGetDeviceCount(&devices) == hipSuccess);
  const char* backend = "hip";
#else
  REQUIRE(cudaGetDeviceCount(&devices) == cudaSuccess);
  const char* backend = "cuda";
#endif
  REQUIRE(devices > 0);
  ScopedComplexGpuControls controls;
  controls.set("PSIFORMER_GPU_KFAC_AFFINE", "1");
  Libxml2Document document;
  REQUIRE(document.parse(input));
  const auto node = [&](const char* expression) -> xmlNodePtr {
    auto* result = xmlXPathEvalExpression(reinterpret_cast<const xmlChar*>(expression), document.getXPathContext());
    REQUIRE(result != nullptr);
    REQUIRE(result->nodesetval != nullptr);
    REQUIRE(result->nodesetval->nodeNr == 1);
    auto* selected = result->nodesetval->nodeTab[0];
    xmlXPathFreeObject(result);
    return selected;
  };
  const auto content = [](xmlNodePtr selected) {
    auto* text = xmlNodeGetContent(selected);
    const std::string value(reinterpret_cast<const char*>(text));
    xmlFree(text);
    return value;
  };
  Lattice lattice;
  std::istringstream lattice_values(content(node("//simulationcell/parameter[@name='lattice']")));
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      REQUIRE(static_cast<bool>(lattice_values >> lattice.R(row, column)));
  REQUIRE(content(node("//simulationcell/parameter[@name='bconds']")).find("p p p") != std::string::npos);
  lattice.BoxBConds = {true, true, true};
  lattice.reset();
  const SimulationCell simulation_cell(lattice);
  auto source = std::make_unique<ParticleSet>(simulation_cell);
  XMLParticleParser source_parser(*source);
  REQUIRE(source_parser.readXML(node("//particleset[@name='ion0']")));
  REQUIRE(source->getTotalNum() == 100);
  source->update();
  ParticleSet p(simulation_cell);
  p.setName("e");
  p.create({50, 50});
  auto& species = p.getSpeciesSet();
  species.addSpecies("u"); species.addSpecies("d");
  const int mass = species.addAttribute("mass");
  species(mass, 0) = species(mass, 1) = 1;
  p.resetGroups();
  for (int electron = 0; electron < 100; ++electron)
    p.R[electron] = source->R[electron] + ParticleSet::PosType{
        .17 + .03 * (electron / 50), .21 + .002 * electron, -.13 + .001 * electron};
  p.update();
  const auto positions = p.R;
  WaveFunctionComponentBuilder::PSetMap sources;
  sources.emplace("ion0", std::move(source));
  auto* head = node("//psiformer");
  const auto property = [&](const char* key, const char* value) {
    xmlSetProp(head, reinterpret_cast<const xmlChar*>(key), reinterpret_cast<const xmlChar*>(value));
  };
  property("optimize", "yes");
  property("optimize_scope", "all");
  property("max_clone_numeric_bytes", "4294967296");
  const auto build = [&](const char* selected_backend) {
    property("gpu", selected_backend);
    if (std::string(selected_backend) == "cpu") xmlUnsetProp(head, BAD_CAST "max_gpu_numeric_bytes");
    else property("max_gpu_numeric_bytes", "8589934592");
    PsiFormerWaveFunctionBuilder builder(OHMMS::Controller, p, sources);
    return builder.buildComponent(head);
  };
  auto cpu_storage = build("cpu");
  auto& cpu = dynamic_cast<PsiFormerComplexWF&>(*cpu_storage);
  const auto registry_for = [](const PsiFormerComplexWF& component) {
    const auto metadata = component.gpuInferenceEnabled() ? component.hybridAffineParameterBlocks() : component.affineParameterBlocks();
    std::vector<KFACAffineBlockDescriptor> blocks;
    std::vector<bool> covered(component.parameterSchema().blocks().size());
    for (const auto& block : metadata)
    {
      blocks.push_back({block.id, block.weight_block, block.bias_block, block.input_width, block.output_width});
      covered.at(block.weight_block) = true;
      if (block.bias_block != NO_KFAC_BIAS_BLOCK) covered.at(block.bias_block) = true;
    }
    std::vector<std::size_t> fallback;
    for (std::size_t i = 0; i < covered.size(); ++i)
      if (!covered[i] && component.parameterSchema().blocks()[i].trainable) fallback.push_back(i);
    return KFACBlockRegistry(component.parameterSchema(), std::move(blocks), std::move(fallback), component.genericKFACParameterBlocks());
  };
  const auto registry = registry_for(cpu);
  KFACFactorAccumulator reference(registry, 0);
  // Independent HOST affine rows form the reference before either GPU route.
  struct Rows final : AffineScoreObserver
  {
    KFACFactorAccumulator& accumulator;
    explicit Rows(KFACFactorAccumulator& a) : accumulator(a) {}
    void observe(const AffineScoreObservation& row) override
    {
      const auto& block = accumulator.registry().affineBlocks().at(row.affine_block);
      accumulator.addObservation({row.affine_block, {row.activations, row.row_count * block.input_width},
          {row.sensitivities, row.row_count * block.output_width}, row.row_count,
          {row.imaginary_sensitivities, row.row_count * block.output_width}});
    }
  } rows(reference);
  cpu.prepareAffineScores(cpu.affineScoreStorage().required_bytes);
  std::vector<double> real, imaginary;
  reference.beginSample(1);
  cpu.observeComplexAffineScores(p, rows, real, imaginary);
  reference.addFallbackScores({real.data(), real.size()}, {imaginary.data(), imaginary.size()});
  reference.endSample();
  reduceKFACFactorStatistics(reference);
  std::vector<double> flat_positions(300);
  for (std::size_t i = 0; i < 100; ++i)
    for (std::size_t d = 0; d < 3; ++d) flat_positions[3*i+d] = p.R[i][d];
  std::vector<cuDoubleComplex> original_score;
  std::vector<KFACFactorStatistics> original_factors;
  std::vector<double> original_fallback;
  // Scan all entries, but keep Catch/reporting overhead proportional to fields.
  const auto compare_field = [](const char* field, const auto& actual,
      const auto& expected, const std::vector<double>* old) {
    REQUIRE(actual.size() == expected.size());
    if (old) REQUIRE(actual.size() == old->size());
    double max_abs = 0, reference_at_max = 0, max_scaled = 0;
    bool finite = true, exact = true;
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
      const double error = std::abs(actual[i] - expected[i]);
      finite = finite && std::isfinite(actual[i]) && std::isfinite(expected[i]);
      if (error > max_abs) { max_abs = error; reference_at_max = expected[i]; }
      max_scaled = std::max(max_scaled, error / (3e-10 + 3e-12 * std::abs(expected[i])));
      if (old) exact = exact && actual[i] == (*old)[i];
    }
    CAPTURE(field, actual.size(), max_abs, reference_at_max, max_scaled);
    CHECK(finite); CHECK(max_scaled <= 1); CHECK(exact);
    std::printf("PAIRED_SHARED_MEANS_FIELD field=%s entries=%zu max_abs=%.17g reference_at_max=%.17g max_scaled=%.17g old_new_exact=%d\n",
        field, actual.size(), max_abs, reference_at_max, max_scaled, exact);
  };
  for (const char* selector : {"0", "1"})
  {
    std::printf("PAIRED_SHARED_MEANS_BEGIN N=100 selector=%s input=%s\n", selector, input);
    controls.set("PSIFORMER_GPU_SOFTMAX_KEY_SHARED_MEANS", selector);
    RuntimeOptions options;
    TrialWaveFunction trial(options);
    auto storage = build(backend);
    auto& gpu = dynamic_cast<PsiFormerComplexWF&>(*storage);
    trial.addComponent(std::move(storage));
    const auto gpu_registry = registry_for(gpu);
    REQUIRE(gpu_registry.fingerprint() == registry.fingerprint());
    const auto* pointer = static_cast<const dev::PreparedDeviceModel*>(gpu.numericStorage().shared_image_identity);
    std::shared_ptr<const dev::PreparedDeviceModel> image(pointer, [](const auto*) {});
    REQUIRE(image->plan().modelShape().electrons() == 100);
    REQUIRE(image->plan().modelShape().feature_dimension >= 50);
    compute::Queue<PlatformKind::CUDA> queue;
    dev::AcceleratorBlasHandle blas(queue);
    auto score = dev::PreparedDeviceComplexScore::prepare(image, queue, blas, 8589934592ULL);
    score->evaluate(flat_positions);
    std::vector<cuDoubleComplex> actual(real.size());
    dev::ComplexValueResult amplitude;
    score->download(score->resultView(), actual, amplitude);
    std::vector<double> actual_real(actual.size()), actual_imaginary(actual.size());
    std::vector<double> old_real(original_score.size()), old_imaginary(original_score.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
    { actual_real[i] = cuCreal(actual[i]); actual_imaginary[i] = cuCimag(actual[i]); }
    for (std::size_t i = 0; i < original_score.size(); ++i)
    { old_real[i] = cuCreal(original_score[i]); old_imaginary[i] = cuCimag(original_score[i]); }
    compare_field("score_real", actual_real, real, old_real.empty() ? nullptr : &old_real);
    compare_field("score_imaginary", actual_imaginary, imaginary, old_imaginary.empty() ? nullptr : &old_imaginary);
    RefVectorWithLeader<TrialWaveFunction> trials(trial); trials.push_back(trial);
    RefVectorWithLeader<ParticleSet> particles(p); particles.push_back(p);
    constexpr auto scope = PsiFormerComplexWF::GpuDerivativeScope::SCORE_ECP_ONLY;
    const auto requirement = gpu.gpuDerivativeRequirements(1, 1, 1, 7, scope);
    auto op = gpu.makeBudgetedGpuDerivativeOperator(trials, particles, 0, 0, 1, 1, 7,
        requirement.device_bytes, requirement.host_bytes, 1, scope);
    const auto factors_requirement = gpu.deviceAffineStatisticsRequirements(gpu_registry);
    KFACFactorAccumulator factors(gpu_registry, 0, KFACCurvatureConvention::NATIVE_SCORE,
        KFACAffineStatisticsSource::PRECOMPUTED_SUMS);
    // Import only complete DEVICE factor transactions into the ordinary ledger.
    struct Sink final : DeviceAffineStatisticsSink
    {
      KFACFactorAccumulator& accumulator;
      explicit Sink(KFACFactorAccumulator& a) : accumulator(a) {}
      void beginSample() override { accumulator.beginSample(1); }
      void addFallbackScore(const std::vector<double>& r) override { accumulator.addFallbackScores({r.data(),r.size()}); }
      void addComplexFallbackScore(const std::vector<double>& r, const std::vector<double>& i) override
      { accumulator.addFallbackScores({r.data(),r.size()}, {i.data(),i.size()}); }
      void endSample() override { accumulator.endSample(); }
      void abortSample() noexcept override { accumulator.abortSample(); }
      void import(std::vector<KFACFactorStatistics> f) override
      { accumulator.importAffineStatistics(accumulator.registry().fingerprint(), 0, std::move(f)); }
    } sink(factors);
    gpu.accumulateDeviceAffineStatistics({&p}, gpu_registry, 0, false, sink,
        factors_requirement.device_bytes + factors_requirement.host_bytes);
    const auto& expected = reference.factors();
    const auto& result = factors.factors();
    REQUIRE(result.size() == expected.size());
    for (std::size_t block = 0; block < result.size(); ++block)
      for (bool activation : {true, false})
      {
        const auto& a = activation ? result[block].activation_outer_sum : result[block].sensitivity_outer_sum;
        const auto& e = activation ? expected[block].activation_outer_sum : expected[block].sensitivity_outer_sum;
        const std::vector<double>* old = nullptr;
        if (!original_factors.empty()) old = activation ? &original_factors[block].activation_outer_sum :
            &original_factors[block].sensitivity_outer_sum;
        compare_field(activation ? "A" : "G", a, e, old);
      }
    const auto fallback = factors.fallbackDiagonalSum(), expected_fallback = reference.fallbackDiagonalSum();
    compare_field("fallback", fallback, expected_fallback, original_fallback.empty() ? nullptr : &original_fallback);
    original_score = actual; original_factors = result;
    original_fallback.assign(fallback.data(), fallback.data() + fallback.size());
    for (std::size_t i = 0; i < p.R.size(); ++i)
      for (int d = 0; d < 3; ++d) CHECK(p.R[i][d] == positions[i][d]);
    std::printf("PAIRED_SHARED_MEANS_COMPLETE N=100 selector=%s P=%zu blocks=%zu two_objectives=1 input=%s\n",
        selector, real.size(), result.size(), input);
  }
}

TEST_CASE("Paired DEVICE affine factors match independent CPU two-channel rows",
          "[wavefunction][psiformer][complex_wf][complex_device_affine]")
{
  using namespace wftrain;
  namespace dev = ps::device;
  int devices = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices) { WARN("NOT_RUN: HIP DEVICE affine"); return; }
  REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) { WARN("NOT_RUN: CUDA DEVICE affine"); return; }
  REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
  ScopedComplexGpuControls controls;
  const std::array<double, 3> twist = GENERATE(std::array<double,3>{0,0,0}, std::array<double,3>{.17,-.11,.07});
  const auto convention = GENERATE(KFACCurvatureConvention::NATIVE_SCORE, KFACCurvatureConvention::REGISTERED_GAUSSIAN);
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell, 1, 3), p1 = electrons(simulation_cell, 1, 3), cpu_p = electrons(simulation_cell, 1, 3);
  p1.R[2][0] += .037; p1.R[1][2] -= .029; p1.update();
  auto source = ions(simulation_cell);
  auto initialized = parameters();
  initialized.model_shape.spin_up_electrons = 1;
  initialized.model_shape.spin_down_electrons = 3;
  RuntimeOptions options;
  TrialWaveFunction trial(options);
  auto owner = std::make_unique<PsiFormerComplexWF>("device_affine", initialized, p, source,
      clone_cap, twist, 64 * 1024 * 1024);
  auto& gpu = *owner;
  trial.addComponent(std::move(owner));
  auto clone = trial.makeClone(p1);
  RefVectorWithLeader<TrialWaveFunction> trials(trial); trials.push_back(trial); trials.push_back(*clone);
  RefVectorWithLeader<ParticleSet> particles(p); particles.push_back(p); particles.push_back(p1);
  PsiFormerComplexWF cpu("cpu_affine", initialized, cpu_p, source, clone_cap, twist);
  const auto make_registry = [](const PsiFormerComplexWF& component) {
    const auto metadata = component.gpuInferenceEnabled() ? component.hybridAffineParameterBlocks() : component.affineParameterBlocks();
    std::vector<KFACAffineBlockDescriptor> blocks;
    std::vector<bool> covered(component.parameterSchema().blocks().size());
    for (const auto& block : metadata)
    {
      blocks.push_back({block.id, block.weight_block, block.bias_block, block.input_width, block.output_width});
      covered.at(block.weight_block) = true;
      if (block.bias_block != NO_KFAC_BIAS_BLOCK) covered.at(block.bias_block) = true;
    }
    std::vector<std::size_t> fallback;
    for (std::size_t i = 0; i < covered.size(); ++i)
      if (!covered[i] && component.parameterSchema().blocks()[i].trainable) fallback.push_back(i);
    return KFACBlockRegistry(component.parameterSchema(), std::move(blocks), std::move(fallback), component.genericKFACParameterBlocks());
  };
  const auto cpu_registry = make_registry(cpu), registry = make_registry(gpu);
  KFACFactorAccumulator expected(cpu_registry, 0, convention);
  struct Rows final : AffineScoreObserver
  {
    KFACFactorAccumulator& accumulator;
    explicit Rows(KFACFactorAccumulator& a) : accumulator(a) {}
    void observe(const AffineScoreObservation& row) override
    {
      const auto& block = accumulator.registry().affineBlocks().at(row.affine_block);
      accumulator.addObservation({row.affine_block, {row.activations, row.row_count * block.input_width},
          {row.sensitivities, row.row_count * block.output_width}, row.row_count,
          {row.imaginary_sensitivities, row.row_count * block.output_width}});
    }
  } rows(expected);
  cpu.prepareAffineScores(cpu.affineScoreStorage().required_bytes);
  std::vector<double> real, imaginary;
  for (const auto* sample : {&p, &p1})
  {
    expected.beginSample(1);
    cpu.observeComplexAffineScores(*sample, rows, real, imaginary);
    expected.addFallbackScores({real.data(), real.size()}, {imaginary.data(), imaginary.size()});
    expected.endSample();
  }
  reduceKFACFactorStatistics(expected);
  CHECK_FALSE(gpu.usesDeviceAffineStatistics());
  controls.set("PSIFORMER_GPU_KFAC_AFFINE", "bad");
  CHECK_THROWS(gpu.usesDeviceAffineStatistics());
  controls.set("PSIFORMER_GPU_KFAC_AFFINE", "1");
  REQUIRE(gpu.usesDeviceAffineStatistics());
  const auto requirement = gpu.deviceAffineStatisticsRequirements(registry);
  const auto numeric = requirement.device_bytes + requirement.host_bytes;
  REQUIRE(requirement.device_bytes > 0); REQUIRE(requirement.host_bytes > 0);
  {
    // Borrow only within gpu's lifetime; no second image or accepted-state mutation.
    const auto* image_pointer = static_cast<const dev::PreparedDeviceModel*>(gpu.numericStorage().shared_image_identity);
    std::shared_ptr<const dev::PreparedDeviceModel> image(image_pointer, [](const auto*) {});
    const auto& tensor = image->plan().parameter(ps::ParameterRole::BACKFLOW_UP_WEIGHT);
    const auto in = tensor.shape[0], out = tensor.shape[1];
    std::vector<dev::DeviceAffineFactorDescriptor> blocks{{ps::ParameterRole::BACKFLOW_UP_WEIGHT,
        ps::NO_ATTENTION_BLOCK, in, out, false}};
    compute::Queue<PlatformKind::CUDA> queue;
    dev::AcceleratorBlasHandle blas(queue);
    using Owner = dev::PreparedDeviceAffineStatistics;
    Owner factors(image, blocks, false, Owner::requiredDeviceBytes(blocks), Owner::requiredHostBytes(blocks), queue, blas);
    std::vector<double> x(in), r(out), im(out);
    for (std::size_t j = 0; j < in; ++j) x[j] = .01 * (j + 1);
    for (std::size_t j = 0; j < out; ++j) { r[j] = .03 * (j + 1); im[j] = .04 * (2.0 - j); }
    dev::DeviceBuffer<double> dx(in), dx_other(in), dr(out), di(out);
    dx.upload(x.data(), in); dx_other.upload(x.data(), in); dr.upload(r.data(), out); di.upload(im.data(), out);
    const auto observe = [&](std::size_t objective, const double* activation) {
      factors.observeChannel(ps::ParameterRole::BACKFLOW_UP_WEIGHT, ps::NO_ATTENTION_BLOCK,
          ps::HeadTensorComponent::REAL, objective, activation, objective ? di.data() : dr.data(), 1, in, out);
    };
    CHECK_THROWS(factors.beginSample(1)); CHECK_THROWS(factors.beginSample(0, .5));
    factors.beginSample(0); CHECK_THROWS(observe(1, dx.data())); factors.abortSample();
    factors.beginSample(0); observe(0, dx.data()); CHECK_THROWS(observe(0, dx.data()));
    CHECK_THROWS(factors.prepareSampleCommit()); CHECK(factors.sampleCount() == 0); factors.abortSample();
    factors.beginSample(0); observe(0, dx.data()); CHECK_THROWS(observe(1, dx_other.data())); factors.abortSample();
    factors.beginSample(0); observe(0, dx.data()); observe(1, dx.data()); factors.finishSample();
    std::vector<dev::DeviceAffineFactorSums> downloaded(1);
    downloaded[0].activation_outer_sum.resize(in * in); downloaded[0].sensitivity_outer_sum.resize(out * out);
    factors.download(downloaded);
    CHECK(downloaded[0].sample_weight_sum == 1); CHECK(downloaded[0].row_count == 1);
    for (std::size_t a = 0; a < in; ++a)
      for (std::size_t b = 0; b < in; ++b) close(downloaded[0].activation_outer_sum[a*in+b], x[a]*x[b], 1e-14);
    for (std::size_t a = 0; a < out; ++a)
      for (std::size_t b = 0; b < out; ++b)
        close(downloaded[0].sensitivity_outer_sum[a*out+b], r[a]*r[b]+im[a]*im[b], 1e-14);
    const auto accepted_factors = downloaded;
    x[0] = 1e308; dx.upload(x.data(), in);
    factors.beginSample(0); observe(0, dx.data()); observe(1, dx.data());
    CHECK_THROWS(factors.prepareSampleCommit()); factors.abortSample();
    factors.download(downloaded);
    CHECK(factors.sampleCount() == 1);
    CHECK(downloaded[0].activation_outer_sum == accepted_factors[0].activation_outer_sum);
    CHECK(downloaded[0].sensitivity_outer_sum == accepted_factors[0].sensitivity_outer_sum);
  }
  struct Sink final : DeviceAffineStatisticsSink
  {
    KFACFactorAccumulator& accumulator;
    bool fail = false, imported = false;
    std::function<void()> after_accept;
    explicit Sink(KFACFactorAccumulator& a) : accumulator(a) {}
    void beginSample() override { accumulator.beginSample(1); }
    void addFallbackScore(const std::vector<double>& r) override { accumulator.addFallbackScores({r.data(),r.size()}); }
    void addComplexFallbackScore(const std::vector<double>& r, const std::vector<double>& i) override
    { accumulator.addFallbackScores({r.data(),r.size()}, {i.data(),i.size()}); }
    void endSample() override
    {
      if (fail) throw std::runtime_error("injected HOST affine acceptance failure");
      accumulator.endSample();
      if (after_accept) after_accept();
    }
    void abortSample() noexcept override { accumulator.abortSample(); }
    void import(std::vector<KFACFactorStatistics> factors) override
    {
      accumulator.importAffineStatistics(accumulator.registry().fingerprint(), accumulator.parameterVersion(), std::move(factors));
      imported = true;
    }
  };
  const std::vector<const ParticleSet*> samples{&p,&p1};
  KFACFactorAccumulator unprepared(registry, 0, convention, KFACAffineStatisticsSource::PRECOMPUTED_SUMS);
  Sink unprepared_sink(unprepared);
  CHECK_THROWS(gpu.accumulateDeviceAffineStatistics(samples, registry, 0, false, unprepared_sink, numeric));
  constexpr auto scope = PsiFormerComplexWF::GpuDerivativeScope::SCORE_ECP_ONLY;
  const auto derivative = gpu.gpuDerivativeRequirements(2, 2, 1, 7, scope);
  auto op = gpu.makeBudgetedGpuDerivativeOperator(trials, particles, 0, 0, 2, 1, 7,
      derivative.device_bytes, derivative.host_bytes, 1, scope);
  const auto fingerprint = op->storageDiagnostics().storage_fingerprint;
  const auto accepted = evaluate(gpu, p);
  const auto positions = p.R;
  const auto gradients = p.G;
  const auto laps = p.L;
  auto& metrics = dev::deviceMemoryMetrics(ordinal);
  const auto before = metrics.snapshot();
  CHECK_THROWS(gpu.accumulateDeviceAffineStatistics(samples, registry, 0, false, unprepared_sink, numeric - 1));
  CHECK(metrics.snapshot().allocation_calls == before.allocation_calls);
  KFACFactorAccumulator rejected(registry, 0, convention, KFACAffineStatisticsSource::PRECOMPUTED_SUMS);
  Sink rejected_sink(rejected); rejected_sink.fail = true;
  CHECK_THROWS(gpu.accumulateDeviceAffineStatistics(samples, registry, 0, false, rejected_sink, numeric));
  CHECK_FALSE(rejected_sink.imported); CHECK(rejected.sampleCount() == 0); CHECK_FALSE(rejected.sampleActive());
  CHECK(metrics.snapshot().live_bytes == before.live_bytes);
  KFACFactorAccumulator changed_source(registry, 0, convention, KFACAffineStatisticsSource::PRECOMPUTED_SUMS);
  Sink changed_sink(changed_source);
  const auto original_source = source.R[0][0];
  changed_sink.after_accept = [&] { source.R[0][0] = original_source + .001; };
  CHECK_THROWS(gpu.accumulateDeviceAffineStatistics(samples, registry, 0, false, changed_sink, numeric));
  CHECK_FALSE(changed_sink.imported);
  CHECK_THROWS(changed_source.factors()); // Completed prefixes are not usable precomputed statistics.
  source.R[0][0] = original_source;
  struct ErrorMaximum
  {
    double absolute = 0, relative_nonzero = 0, zero_expected_absolute = 0;
    double reference_at_max_absolute = 0;
    void observe(double actual, double expected)
    {
      const double error = std::abs(actual - expected);
      // Explicitly bound real factors; do not inherit Catch::Approx's relative default.
      CHECK(error <= 3e-10 + 3e-12 * std::abs(expected));
      if (error > absolute)
      {
        absolute = error;
        reference_at_max_absolute = std::abs(expected);
      }
      if (expected != 0) relative_nonzero = std::max(relative_nonzero, error / std::abs(expected));
      else zero_expected_absolute = std::max(zero_expected_absolute, error);
    }
  } a_error, g_error, fallback_error;
  for (int repeat = 0; repeat < 2; ++repeat)
  {
    KFACFactorAccumulator actual(registry, 0, convention, KFACAffineStatisticsSource::PRECOMPUTED_SUMS);
    Sink sink(actual);
    gpu.accumulateDeviceAffineStatistics(samples, registry, 0,
        convention == KFACCurvatureConvention::REGISTERED_GAUSSIAN, sink, numeric);
    REQUIRE(sink.imported);
    reduceKFACFactorStatistics(actual);
    CHECK(actual.sampleCount() == 2);
    CHECK(actual.fallbackWeightSum() == expected.fallbackWeightSum());
    REQUIRE(actual.factors().size() == expected.factors().size());
    for (std::size_t b = 0; b < actual.factors().size(); ++b)
    {
      const auto& a = actual.factors()[b]; const auto& e = expected.factors()[b];
      CHECK(a.row_count == e.row_count); CHECK(a.row_weight_sum == e.row_weight_sum);
      CHECK(a.sample_weight_sum == e.sample_weight_sum);
      for (std::size_t i = 0; i < a.activation_outer_sum.size(); ++i)
      {
        a_error.observe(a.activation_outer_sum[i], e.activation_outer_sum[i]);
      }
      for (std::size_t i = 0; i < a.sensitivity_outer_sum.size(); ++i)
      {
        g_error.observe(a.sensitivity_outer_sum[i], e.sensitivity_outer_sum[i]);
      }
    }
    const auto a = actual.fallbackDiagonalSum(), e = expected.fallbackDiagonalSum();
    REQUIRE(a.size() == e.size());
    for (std::size_t i = 0; i < a.size(); ++i)
    {
      fallback_error.observe(a[i], e[i]);
    }
    CHECK(metrics.snapshot().live_bytes == before.live_bytes);
    CHECK(op->storageDiagnostics().storage_fingerprint == fingerprint);
    CHECK(gpu.get_log_value() == accepted);
    for (std::size_t electron = 0; electron < p.R.size(); ++electron)
    {
      for (std::size_t d = 0; d < 3; ++d)
      {
        CHECK(p.R[electron][d] == positions[electron][d]);
        CHECK(p.G[electron][d] == gradients[electron][d]);
      }
      CHECK(p.L[electron] == laps[electron]);
    }
    CHECK(gpu.affineScoreStorage().prepared_bytes == 0);
  }
  const auto original = source.R[0][0]; source.R[0][0] += .001;
  CHECK_THROWS(gpu.accumulateDeviceAffineStatistics(samples, registry, 0, false, unprepared_sink, numeric));
  source.R[0][0] = original;
  CHECK_THROWS(gpu.accumulateDeviceAffineStatistics(samples, registry, 1, false, unprepared_sink, numeric));
  CHECK_FALSE(unprepared_sink.imported);
  std::printf("COMPLEX_DEVICE_AFFINE_ERROR absolute_threshold=3e-10 relative_threshold=3e-12 gamma=%d gaussian=%d "
      "A_abs=%.17g A_relative_nonzero=%.17g A_zero_expected_abs=%.17g A_reference_at_max_abs=%.17g "
      "G_abs=%.17g G_relative_nonzero=%.17g G_zero_expected_abs=%.17g G_reference_at_max_abs=%.17g "
      "fallback_abs=%.17g fallback_relative_nonzero=%.17g fallback_zero_expected_abs=%.17g fallback_reference_at_max_abs=%.17g\n",
      twist == std::array<double,3>{0,0,0}, convention == KFACCurvatureConvention::REGISTERED_GAUSSIAN,
      a_error.absolute, a_error.relative_nonzero, a_error.zero_expected_absolute, a_error.reference_at_max_absolute,
      g_error.absolute, g_error.relative_nonzero, g_error.zero_expected_absolute, g_error.reference_at_max_absolute,
      fallback_error.absolute, fallback_error.relative_nonzero, fallback_error.zero_expected_absolute,
      fallback_error.reference_at_max_absolute);
  std::printf("COMPLEX_DEVICE_AFFINE PASS Gamma_or_twist imbalanced1_3 all_A_G_fallback two_samples both_conventions HOSTrollback existing_owner\n");
}
#endif

TEST_CASE("Complex provider affine rows preserve canonical phase scores and bounded factors",
          "[wavefunction][psiformer][complex_wf][complex_affine_provider]")
{
  using namespace wftrain;
  bool pair_profile = false;
  bool occupied_phase = false;
  bool polynomial = false;
  bool fourier = false;
  SECTION("legacy paired profile") {}
  SECTION("two-cell pair profile with complete physical ions") { pair_profile = true; }
  SECTION("two-cell occupied reciprocal profile with complete physical ions")
  {
    pair_profile = true;
    occupied_phase = true;
  }
  SECTION("polynomial two-cell occupied profile with complete physical ions")
  {
    pair_profile = true;
    occupied_phase = true;
    polynomial = true;
  }
  SECTION("Fourier replacement two-cell occupied profile with complete physical ions")
  {
    pair_profile = true;
    occupied_phase = true;
    fourier = true;
  }
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell);
  auto source = [&]() {
    if (!pair_profile) return ions(simulation_cell);
    ParticleSet result(simulation_cell);
    result.setName("ion0");
    result.create({2, 2});
    auto& species = result.getSpeciesSet();
    species.addSpecies("Li"); species.addSpecies("H");
    const int charge = species.addAttribute("charge");
    species(charge, 0) = 3; species(charge, 1) = 1;
    result.resetGroups();
    result.R[0] = {0, 0, 0}; result.R[1] = {4, 0, 0};
    result.R[2] = {1, 1, 1}; result.R[3] = {5, 1, 1};
    result.update();
    return result;
  }();
  auto initialized = parameters();
  if (polynomial)
  {
    auto embedding = std::find_if(initialized.layouts.begin(), initialized.layouts.end(), [](const auto& layout) {
      return layout.module.find("/electron_embedding/linear") != std::string::npos;
    });
    const auto end = embedding->end, width = initialized.model_shape.feature_dimension;
    const auto rows = 4 * initialized.model_shape.nuclei + 1;
    const auto removed = embedding->end - embedding->begin - rows * width;
    initialized.values.erase(initialized.values.begin() + embedding->begin + rows * width,
                             initialized.values.begin() + end);
    embedding->shape[0] = rows; embedding->end -= removed;
    for (auto& layout : initialized.layouts)
      if (layout.begin >= end) { layout.begin -= removed; layout.end -= removed; }
  }
  if (pair_profile)
  {
    auto& environment = initialized.environment;
    environment.boundary = ps::BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = ps::GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1;
    environment.head_representation = ps::HeadRepresentation::PAIRED_REAL_COMPLEX;
    environment.amplitude_scalar_domain = ps::ScalarDomain::COMPLEX;
    environment.periodic_axes = {true, true, true};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) environment.lattice_vectors[i][j] = cell().R(i, j);
    auto metadata = std::make_shared<ps::PeriodicProfileMetadata>();
    metadata->cells.primitive = metadata->cells.simulation =
        {pf::GeometryBoundaryKind::PERIODIC, environment.lattice_vectors, environment.periodic_axes};
    metadata->cells.primitive.lattice_vectors[0][0] = 4;
    metadata->cells.tiling[0][0] = 2;
    metadata->primitive_motif = {{0, 0, 0}, {1, 1, 1}};
    metadata->motif_species = {0, 1};
    metadata->electron_electron_cusp = false;
    metadata->feature_transform = polynomial ? ps::PeriodicFeatureTransform::POLYNOMIAL_V1 :
                                              ps::PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1;
    metadata->pair_feature_dimension = ps::periodicPairFeatureArity(metadata->feature_transform);
    if (fourier)
    {
      metadata->envelope_policy = ps::PeriodicEnvelopePolicy::FOURIER;
      metadata->fourier_modes = {{0, 0, 0}, {1, -1, 0}, {0, 1, 2}};
      metadata->fourier_origin = {2.5, .5, .5};
    }
    for (std::size_t i = 0; i < 4; ++i) metadata->occupied_columns.push_back({{}, i % 2, int(i / 2), 1});
    if (occupied_phase)
    {
      const std::array<std::array<int, 3>, 4> momenta{{{-1, 0, 0}, {0, 1, 0}, {1, 0, 1}, {0, -1, -1}}};
      for (std::size_t i = 0; i < 4; ++i) metadata->occupied_columns[i].reciprocal = momenta[i];
    }
    environment.periodic_profile = std::move(metadata);
    const auto pair_arity = environment.periodic_profile->pair_feature_dimension;
    initialized.profile = ps::PSIFORMER_PERIODIC_PAIR_V1;
    for (std::size_t block = 0; block < initialized.model_shape.attention_blocks; ++block)
      for (bool bias : {false, true})
      {
        const auto begin = initialized.values.size();
        const auto heads = initialized.model_shape.attention_heads;
        initialized.values.resize(begin + (bias ? 1 : pair_arity) * heads, .02);
        const auto layer = block == 0 ? std::string("electron_gnn_layer") : "electron_gnn_layer_" + std::to_string(block);
        initialized.layouts.push_back({"periodic/" + layer + "/pair_attention", bias ? "b" : "w",
            bias ? std::vector<std::size_t>{heads} : std::vector<std::size_t>{pair_arity, heads}, begin, initialized.values.size()});
      }
  }
  if (fourier)
  {
    std::vector<double> values;
    std::vector<ps::ParameterLayoutInput> layouts;
    for (auto layout : initialized.layouts)
      if (layout.module.find("/exponential_envelopes") == std::string::npos)
      {
        const auto begin = values.size();
        values.insert(values.end(), initialized.values.begin() + layout.begin, initialized.values.begin() + layout.end);
        layout.begin = begin;
        layout.end = values.size();
        layouts.push_back(std::move(layout));
      }
    const auto& shape = initialized.model_shape;
    const auto channels = shape.determinants * shape.electrons();
    for (bool up : {false, true})
    {
      const auto begin = values.size();
      for (std::size_t mode = 0; mode < 3; ++mode)
        for (std::size_t channel = 0; channel < channels; ++channel)
          values.push_back(mode == 0 ? 1 : .03 * std::sin(double(channel + 1 + mode + up)));
      layouts.push_back({"periodic/fourier_envelopes", up ? "coefficients_up" : "coefficients_down",
          {3, shape.determinants, shape.electrons()}, begin, values.size()});
    }
    initialized.values = std::move(values);
    initialized.layouts = std::move(layouts);
    auto invalid = initialized;
    auto metadata = std::make_shared<ps::PeriodicProfileMetadata>(*initialized.environment.periodic_profile);
    invalid.environment.periodic_profile = metadata;
    // The feature motif mean is not the declared full physical-source image mean.
    metadata->fourier_origin = {.5, .5, .5};
    CHECK_THROWS(PsiFormerComplexWF("bad_origin", invalid, p, source, clone_cap, {.17, -.11, .07}));
  }
  PsiFormerComplexWF component("affine", initialized, p, source, clone_cap, {.17, -.11, .07});
  const auto& schema = component.parameterSchema();
  const auto affine = component.affineParameterBlocks();
  REQUIRE_FALSE(affine.empty());
  std::vector<KFACAffineBlockDescriptor> descriptors;
  std::vector<bool> covered(schema.blocks().size());
  for (const auto& block : affine)
  {
    descriptors.push_back({block.id, block.weight_block, block.bias_block, block.input_width, block.output_width});
    covered[block.weight_block] = true;
    if (block.bias_block != NO_AFFINE_BIAS_BLOCK)
      covered[block.bias_block] = true;
  }
  std::vector<std::size_t> fallback;
  for (std::size_t block = 0; block < covered.size(); ++block)
    if (!covered[block] && schema.blocks()[block].trainable)
      fallback.push_back(block);
  KFACFactorAccumulator factors({schema, descriptors, fallback, component.genericKFACParameterBlocks()}, 0,
                                 KFACCurvatureConvention::REGISTERED_GAUSSIAN);
  class Observer final : public AffineScoreObserver
  {
  public:
    Observer(const StructuredParameterSchema& schema, const std::vector<AffineParameterBlock>& blocks,
             KFACFactorAccumulator& factors)
        : response(schema.parameterCount()), schema_(schema), blocks_(blocks), factors_(factors) {}
    void observe(const AffineScoreObservation& observation) override
    {
      REQUIRE(observation.imaginary_sensitivities != nullptr);
      const auto& block = blocks_.at(observation.affine_block);
      const auto weight_offset = schema_.blocks()[block.weight_block].offset;
      for (std::size_t row = 0; row < observation.row_count; ++row)
        for (std::size_t output = 0; output < block.output_width; ++output)
        {
          const auto index = row * block.output_width + output;
          const C sensitivity{observation.sensitivities[index], observation.imaginary_sensitivities[index]};
          for (std::size_t input = 0; input < block.input_width; ++input)
            response[weight_offset + input * block.output_width + output] +=
                observation.activations[row * block.input_width + input] * sensitivity;
          if (block.bias_block != NO_AFFINE_BIAS_BLOCK)
            response[schema_.blocks()[block.bias_block].offset + output] += sensitivity;
        }
      factors_.addObservation({observation.affine_block,
          {observation.activations, observation.row_count * block.input_width},
          {observation.sensitivities, observation.row_count * block.output_width}, observation.row_count,
          {observation.imaginary_sensitivities, observation.row_count * block.output_width}});
    }
    std::vector<C> response;
  private:
    const StructuredParameterSchema& schema_;
    const std::vector<AffineParameterBlock>& blocks_;
    KFACFactorAccumulator& factors_;
  } observer(schema, affine, factors);
  std::vector<double> real, imaginary;
  CHECK_THROWS(component.observeComplexAffineScores(p, observer, real, imaginary));
  const auto required = component.affineScoreStorage().required_bytes;
  CHECK(component.affineScoreStorage().prepared_bytes == 0);
  CHECK_THROWS(component.hybridAffineParameterBlocks());
  CHECK_THROWS(component.prepareHybridAffineScores(required));
  CHECK_THROWS(component.observeHybridComplexAffineScores(p, observer, real, imaginary));
  CHECK(component.affineScoreStorage().prepared_bytes == 0);
  CHECK_THROWS(component.prepareAffineScores(required - 1));
  CHECK(component.affineScoreStorage().prepared_bytes == 0);
  component.prepareAffineScores(required);
  CHECK(component.affineScoreStorage().prepared_bytes == required);
  factors.beginSample(1);
  component.observeComplexAffineScores(p, observer, real, imaginary);
  factors.addFallbackScores({real.data(), real.size()}, {imaginary.data(), imaginary.size()});
  factors.endSample();
  reduceKFACFactorStatistics(factors);
  REQUIRE(real.size() == schema.parameterCount());
  REQUIRE(imaginary.size() == real.size());
  bool phase_response = false;
  for (std::size_t block = 0; block < covered.size(); ++block)
    if (covered[block])
      for (std::size_t index = schema.blocks()[block].offset;
           index < schema.blocks()[block].offset + schema.blocks()[block].count; ++index)
      {
        close(observer.response[index], {real[index], imaginary[index]}, 3e-10);
        phase_response |= std::abs(imaginary[index]) > 1e-8;
      }
  CHECK(phase_response);
  CHECK(factors.sampleCount() == 1);
  const KFACPreconditioner inverse(factors);
  CHECK(inverse.storageDiagnostics().affine_block_count == affine.size());
  const auto original = component.snapshotParameters();
  const double step = 1e-5;
  for (const auto& block : affine)
  {
    const auto index = schema.blocks()[block.weight_block].offset;
    auto candidate = component.snapshotParameters();
    candidate.values[index] += step;
    component.publishParameters(candidate, candidate.version);
    const C plus = evaluate(component, p);
    candidate = component.snapshotParameters();
    candidate.values[index] -= 2 * step;
    component.publishParameters(candidate, candidate.version);
    const C minus = evaluate(component, p);
    candidate = component.snapshotParameters();
    candidate.values = original.values;
    component.publishParameters(candidate, candidate.version);
    close({real[index], imaginary[index]}, (plus - minus) / (2 * step), 2e-7);
  }
  CHECK(component.affineScoreStorage().prepared_bytes == required);
  if (fourier)
  {
    std::vector<C> raw_values;
    component.evaluateOrbitalValues(p, raw_values, component.snapshotParameters().version);
    const std::vector<C> target(raw_values.size(), C{.02, -.03});
    auto value_loss = [&]() {
      component.evaluateOrbitalValues(p, raw_values, component.snapshotParameters().version);
      double loss = 0;
      for (std::size_t index = 0; index < target.size(); ++index)
        loss += std::norm(raw_values[index] - target[index]) / (component.orbitalMSEChannelCount() * 2 * 4);
      return loss;
    };
    const auto owner = component.orbitalMSEStorage();
    CHECK(owner.prepared_bytes == 0);
    CHECK_THROWS(component.prepareOrbitalMSE(owner.required_bytes - 1));
    component.prepareOrbitalMSE(owner.required_bytes);
    CHECK(component.orbitalMSEStorage().prepared_bytes == owner.required_bytes);
    std::vector<double> gradient(original.values.size());
    const auto version = component.snapshotParameters().version;
    const double loss = component.evaluateOrbitalMSE(p, target, gradient, version);
    CHECK(loss == Catch::Approx(value_loss()).epsilon(1e-12));
    const ps::PsiFormerExecutionPlan plan(initialized.model_shape, initialized.layouts, initialized.environment);
    const auto generic = component.genericKFACParameterBlocks();
    for (auto role : {ps::ParameterRole::FOURIER_COEFFICIENT_UP, ps::ParameterRole::FOURIER_COEFFICIENT_DOWN})
    {
      const auto& tensor = plan.parameter(role);
      const auto block = std::find_if(schema.blocks().begin(), schema.blocks().end(), [&](const auto& candidate) {
        return candidate.offset == tensor.begin;
      });
      REQUIRE(block != schema.blocks().end());
      CHECK(std::find(generic.begin(), generic.end(), std::size_t(block - schema.blocks().begin())) != generic.end());
      const auto index = tensor.begin + initialized.model_shape.determinants * initialized.model_shape.electrons();
      auto candidate = component.snapshotParameters();
      candidate.values[index] += step;
      component.publishParameters(candidate, candidate.version);
      const C plus = evaluate(component, p);
      const double plus_loss = value_loss();
      candidate = component.snapshotParameters();
      candidate.values[index] -= 2 * step;
      component.publishParameters(candidate, candidate.version);
      const C minus = evaluate(component, p);
      const double minus_loss = value_loss();
      candidate = component.snapshotParameters();
      candidate.values = original.values;
      component.publishParameters(candidate, candidate.version);
      close({real[index], imaginary[index]}, (plus - minus) / (2 * step), 2e-7);
      CHECK(gradient[index] == Catch::Approx((plus_loss - minus_loss) / (2 * step)).margin(2e-8));
    }
  }
  if (pair_profile)
  {
    evaluate(component, p);
    C expected_kinetic{};
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
    {
      C contraction = p.L[electron];
      for (int d = 0; d < 3; ++d) contraction += p.G[electron][d] * p.G[electron][d];
      expected_kinetic -= .5 * contraction;
    }
    close(component.observeCoulombEnergy(p, 0).kinetic, expected_kinetic);
    RefVectorWithLeader<WaveFunctionComponent> components(component);
    components.push_back(component);
    RefVectorWithLeader<ParticleSet> particles(p);
    particles.push_back(p);
    const auto derivative_bytes = component.streamingDerivativeRequiredBytes(1);
    CHECK_THROWS(component.makeBudgetedStreamingDerivativeOperator(components, particles, 0, 0, 7,
                                                                  derivative_bytes - 1));
    auto derivative = component.makeBudgetedStreamingDerivativeOperator(components, particles, 0, 0, 7,
                                                                       derivative_bytes);
    CHECK(derivative->storageDiagnostics().retained_numeric_bytes == derivative_bytes);
    // The second replica is not a feature center, but remains fixed physical data.
    const auto fixed = source.R[1][0];
    source.R[1][0] += .01;
    CHECK_THROWS(component.observeComplexAffineScores(p, observer, real, imaginary));
    source.R[1][0] = fixed;
    auto clone_particles = electrons(simulation_cell);
    auto clone = component.makeClone(clone_particles);
    auto* typed = dynamic_cast<PsiFormerComplexWF*>(clone.get());
    REQUIRE(typed);
    typed->isolateParameterState();
    close(evaluate(*typed, clone_particles), evaluate(component, p));
    CHECK(typed->trainingConfigurationFingerprint() == component.trainingConfigurationFingerprint());
    const auto path = "psiformer_pair_component." + std::to_string(OHMMS::Controller->rank()) + ".h5";
    REQUIRE_FALSE(std::filesystem::exists(path));
    component.exportModel(path);
    auto imported = ps::readPsiFormerModelFile(path);
    REQUIRE(imported.parameters.environment.periodic_profile);
    CHECK(imported.parameters.model_shape.nuclei == 2);
    const auto& metadata = *imported.parameters.environment.periodic_profile;
    CHECK(metadata.cells.tiling[0][0] == 2);
    CHECK(metadata.feature_transform == initialized.environment.periodic_profile->feature_transform);
    CHECK(metadata.pair_feature_dimension == (polynomial ? 5 : 8));
    CHECK_FALSE(metadata.electron_electron_cusp);
    CHECK(metadata.occupied_columns.size() == 4);
    CHECK(metadata.envelope_policy == (fourier ? ps::PeriodicEnvelopePolicy::FOURIER : ps::PeriodicEnvelopePolicy::ATOM_CENTERED));
    if (fourier)
    {
      CHECK(metadata.fourier_modes == initialized.environment.periodic_profile->fourier_modes);
      CHECK(metadata.fourier_origin == std::array<double, 3>{2.5, .5, .5});
      const ps::PsiFormerExecutionPlan plan(imported.parameters.model_shape, imported.parameters.layouts,
                                           imported.parameters.environment);
      CHECK_FALSE(plan.hasParameter(ps::ParameterRole::ENVELOPE_PI_UP));
      CHECK_FALSE(plan.hasParameter(ps::ParameterRole::ENVELOPE_ZETA_DOWN));
      CHECK(plan.parameter(ps::ParameterRole::FOURIER_COEFFICIENT_UP).shape ==
            std::vector<std::size_t>{3, initialized.model_shape.determinants, 4});
    }
    for (std::size_t column = 0; column < metadata.occupied_columns.size(); ++column)
      CHECK(metadata.occupied_columns[column].reciprocal ==
            initialized.environment.periodic_profile->occupied_columns[column].reciprocal);
    ps::validatePsiFormerModelFile(imported, path, clone_particles, source, "all_electron", {.17, -.11, .07});
    PsiFormerComplexWF reloaded("affine", imported.parameters, clone_particles, source, clone_cap,
                               {.17, -.11, .07});
    CHECK(reloaded.snapshotParameters().values == component.snapshotParameters().values);
    CHECK(reloaded.trainingConfigurationFingerprint() == component.trainingConfigurationFingerprint());
    close(evaluate(reloaded, clone_particles), evaluate(component, p));
    source.R[1][0] += .01;
    CHECK_THROWS(ps::validatePsiFormerModelFile(imported, path, clone_particles, source, "all_electron",
                                               {.17, -.11, .07}));
    source.R[1][0] = fixed;
    source.GroupID[1] = 1;
    CHECK_THROWS(ps::validatePsiFormerModelFile(imported, path, clone_particles, source, "all_electron",
                                               {.17, -.11, .07}));
    source.GroupID[1] = 0;
    std::filesystem::remove(path);
  }
}

TEST_CASE("PsiFormer complex structured provider publishes canonical real tensors atomically",
          "[wavefunction][psiformer][complex_wf][complex_provider]")
{
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell);
  auto clone_p = electrons(simulation_cell);
  auto reference_p = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  auto initialized = parameters();
  PsiFormerComplexWF component("provider", initialized, p, source, clone_cap, {.17, -.11, .07});
  auto base_clone = component.makeClone(clone_p);
  auto& clone = dynamic_cast<PsiFormerComplexWF&>(*base_clone);
  CHECK(component.structuredParameterProvider() == &component); // CPU structured route only; scalar optimizer stays closed
  CHECK_FALSE(component.isOptimizable()); // no P scalar names or legacy optimizer admission
  const auto& schema = component.parameterSchema();
  REQUIRE(schema.blocks().size() == initialized.layouts.size());
  REQUIRE(schema.parameterCount() == initialized.values.size());
  CHECK(&schema == &clone.parameterSchema());
  std::size_t imaginary_blocks = 0, changed_index = schema.parameterCount();
  for (std::size_t i = 0; i < schema.blocks().size(); ++i)
  {
    const auto& block = schema.blocks()[i];
    const auto& layout = initialized.layouts[i];
    CHECK(block.id == layout.module + "/" + layout.name);
    CHECK(block.shape == layout.shape);
    CHECK(block.offset == layout.begin);
    CHECK(block.count == layout.end - layout.begin);
    CHECK(block.scalar_domain == wftrain::ParameterScalarDomain::REAL64);
    if (layout.name == "w_imag")
    {
      ++imaginary_blocks;
      changed_index = block.offset;
    }
  }
  CHECK(imaginary_blocks == 2);
  REQUIRE(changed_index < schema.parameterCount());
  const auto original = component.snapshotParameters();
  CHECK(original.values == initialized.values);
  CHECK(clone.snapshotParameters().values == original.values);
  const auto before = evaluate(component, p);
  evaluate(clone, clone_p);
  WaveFunctionComponent::WFBufferType buffer;
  component.registerData(p, buffer);
  buffer.allocate();
  buffer.rewind();
  p.G = 0;
  p.L = 0;
  component.updateBuffer(p, buffer);
  const auto storage = component.numericStorage();

  auto candidate = original;
  candidate.values[changed_index] += .04; // a real canonical coordinate in an independent imaginary head
  const auto committed = component.publishParameters(candidate, original.version);
  CHECK(committed == original.version + 1);
  CHECK(component.snapshotParameters().values == candidate.values);
  CHECK(clone.snapshotParameters().version == committed);
  CHECK(clone.snapshotParameters().values == candidate.values);
  CHECK_THROWS(component.observeCoulombEnergy(p, -2.7));
  CHECK_THROWS(clone.observeCoulombEnergy(clone_p, -2.7));
  buffer.rewind();
  CHECK_THROWS(clone.copyFromBuffer(clone_p, buffer)); // old version is not a current accepted record
  initialized.values = candidate.values;
  PsiFormerComplexWF reference("reference", initialized, reference_p, source, clone_cap, {.17, -.11, .07});
  const auto after = evaluate(component, p);
  close(std::exp(after - evaluate(reference, reference_p)), C{1});
  close(std::exp(after - evaluate(clone, clone_p)), C{1});
  CHECK(std::abs(std::exp(after - before) - C{1}) > 1e-8);
  CHECK(component.numericStorage().allocated_bytes == storage.allocated_bytes);
  CHECK(component.numericStorage().fingerprint == storage.fingerprint);

  const auto baseline = component.snapshotParameters();
  const auto unchanged = [&] {
    const auto snapshot = component.snapshotParameters();
    CHECK(snapshot.version == baseline.version);
    CHECK(snapshot.values == baseline.values);
    CHECK(clone.snapshotParameters().version == baseline.version);
  };
  auto invalid = baseline;
  invalid.schema_fingerprint += "/foreign";
  CHECK_THROWS(component.publishParameters(invalid, baseline.version));
  unchanged();
  invalid = baseline;
  invalid.values.pop_back();
  CHECK_THROWS(component.publishParameters(invalid, baseline.version));
  unchanged();
  for (double nonfinite : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
  {
    invalid = baseline;
    invalid.values[changed_index] = nonfinite;
    CHECK_THROWS(component.publishParameters(invalid, baseline.version));
    unchanged();
  }
  invalid = baseline;
  invalid.version += 1;
  CHECK_THROWS(component.publishParameters(invalid, baseline.version));
  unchanged();
  CHECK_THROWS(component.publishParameters(candidate, original.version)); // formerly valid but now stale
  unchanged();
  invalid = baseline;
  invalid.version = std::numeric_limits<std::size_t>::max();
  CHECK_THROWS(component.publishParameters(invalid, invalid.version));
  unchanged();
  CHECK(component.structuredParameterProvider() == &component);
}

TEST_CASE("PsiFormer complex pending family guard releases on every revocation boundary",
          "[wavefunction][psiformer][complex_wf][complex_provider]")
{
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell);
  auto clone_p = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  PsiFormerComplexWF component("pending", parameters(), p, source, clone_cap);
  auto base_clone = component.makeClone(clone_p);
  auto& clone = dynamic_cast<PsiFormerComplexWF&>(*base_clone);
  const ParticleSet::PosType displacement{.007, -.011, .013};
  const auto publish = [&] {
    const auto snapshot = component.snapshotParameters();
    CHECK(component.publishParameters(snapshot, snapshot.version) == snapshot.version + 1);
  };
  const auto blocked = [&] {
    const auto snapshot = component.snapshotParameters();
    CHECK_THROWS(component.publishParameters(snapshot, snapshot.version));
    CHECK(component.snapshotParameters().version == snapshot.version);
    CHECK(component.snapshotParameters().values == snapshot.values);
  };

  evaluate(clone, clone_p);
  clone_p.makeMove(0, displacement);
  clone.ratio(clone_p, 0);
  blocked(); // a different clone's proposal still blocks the root provider
  clone.restore(0);
  clone_p.rejectMove(0);
  publish();
  evaluate(clone, clone_p);
  clone_p.makeMove(0, displacement);
  clone.ratio(clone_p, 0);
  CHECK_THROWS(clone.restore(1)); // malformed restore also revokes its proposal count
  clone_p.rejectMove(0);
  publish();
  evaluate(clone, clone_p);
  clone_p.makeMove(0, displacement);
  clone.ratio(clone_p, 0);
  CHECK_THROWS(clone.ratio(clone_p, 1)); // failed replacement revokes preceding pending state
  clone_p.rejectMove(0);
  publish();
  evaluate(clone, clone_p);
  clone_p.makeMove(0, displacement);
  clone.ratio(clone_p, 0);
  CHECK_THROWS(clone.acceptMove(clone_p, 1));
  clone_p.rejectMove(0);
  publish();
  evaluate(clone, clone_p);
  clone_p.makeMove(0, displacement);
  clone.ratio(clone_p, 0);
  clone.acceptMove(clone_p, 0);
  clone_p.acceptMove(0);
  publish();
  evaluate(clone, clone_p);
  clone_p.makeMove(0, displacement);
  clone.ratio(clone_p, 0);
  blocked();
  base_clone.reset(); // destruction must not leave the shared family permanently blocked
  clone_p.rejectMove(0);
  publish();
  CHECK_NOTHROW(evaluate(component, p));
}

namespace
{
/// Test-only dense output collector; production retains no sample-by-parameter arrays.
class ComplexStreamingSink final : public wftrain::ParameterReductionSink
{
public:
  std::vector<std::vector<C>> values;
  bool fail_next = false;
  std::function<void()> callback;

  /// Allocate only test oracle storage after common request preflight.
  void onBegin(const wftrain::DerivativeStreamDescriptor&, const wftrain::ParameterChunkPlan& plan,
      wftrain::DerivativeArrayView<const wftrain::VJPCoefficientChannel> channels) override
  {
    std::size_t count = 0;
    for (const auto& chunk : plan.chunks())
      count = std::max(count, chunk.parameter_offset + chunk.count);
    values.assign(channels.size(), std::vector<C>(count));
  }

  /// Copy all checked canonical chunks and support one injected transactional failure.
  void consume(std::size_t channel, const wftrain::ParameterChunkConstView& chunk) override
  {
    if (fail_next)
    {
      fail_next = false;
      throw std::runtime_error("injected complex stream sink failure");
    }
    if (callback)
      callback();
    std::copy(chunk.values().begin(), chunk.values().end(),
              values[channel].begin() + chunk.descriptor().parameter_offset);
  }
};

/// Test-only sample collector supports abort injection and callbacks that require an unlocked model.
class ComplexSampleSink final : public wftrain::SampleProductSink
{
public:
  std::vector<C> values;
  bool fail_next = false;
  std::function<void()> callback;

  /// Prepare one scalar per sample, never a score matrix.
  void onBegin(const wftrain::DerivativeStreamDescriptor& descriptor) override
  { values.assign(descriptor.sample_count, C{}); }

  /// Observe checked tiles only after the producer has completed and released its model read transaction.
  void consume(const wftrain::SampleProductTileConstView& tile) override
  {
    if (fail_next)
    {
      fail_next = false;
      throw std::runtime_error("injected complex sample sink failure");
    }
    if (callback)
      callback();
    const auto offset = tile.descriptor().sample_offset - activeDescriptor()->sample_offset;
    std::copy(tile.values().begin(), tile.values().end(), values.begin() + offset);
  }
};

#ifdef ENABLE_CUDA
/// Test-only immutable accepted-grid record; stamps come from the actual paired VALUE transaction.
struct ComplexCapturedReplay final : wftrain::NonLocalECPJVPReplay
{
  explicit ComplexCapturedReplay(const wftrain::StreamingDerivativeOperator& op)
      : provider(op.parameterSchema().providerId()), schema(op.parameterSchema().fingerprint()),
        version(op.parameterVersion()), ordinal(op.batchOrdinal()), offset(op.sampleOffset()), bases(op.sampleCount())
  {}

  std::string_view providerId() const noexcept override { return provider; }

  std::string_view schemaFingerprint() const noexcept override { return schema; }

  std::size_t parameterVersion() const noexcept override { return version; }

  std::size_t batchOrdinal() const noexcept override { return ordinal; }

  std::size_t sampleOffset() const noexcept override { return offset; }

  std::size_t sampleCount() const noexcept override { return bases; }

  std::size_t electronCount() const noexcept override { return 4; }

  std::uint32_t localEnergyTermMask() const noexcept override
  { return wftrain::localEnergyTermBit(wftrain::LocalEnergyTerm::NONLOCAL_ECP); }

  std::uint64_t gridFingerprint() const noexcept override { return 123; }

  wftrain::DerivativeArrayView<const double> referencePositions() const noexcept override
  { return {positions.data(), positions.size()}; }

  wftrain::DerivativeArrayView<const wftrain::NonLocalECPReplayPoint> points() const noexcept override
  { return {grid.data(), grid.size()}; }

  wftrain::DerivativeArrayView<const wftrain::NonLocalECPReplaySegment> segments() const noexcept override
  { return {segments_.data(), segments_.size()}; }

  wftrain::DerivativeArrayView<const wftrain::NonLocalECPReplayTile> tiles() const noexcept override
  { return {tiles_.data(), tiles_.size()}; }

  wftrain::DerivativeArrayView<const WaveFunctionComponent::EvaluationStamp> evaluationStamps() const noexcept override
  { return {stamps.data(), stamps.size()}; }

  wftrain::NonLocalECPReplayStorageDiagnostics storageDiagnostics() const noexcept override
  {
    wftrain::NonLocalECPReplayStorageDiagnostics d;
    d.point_count = grid.size(); d.segment_count = segments_.size();
    d.tile_count = tiles_.size(); d.evaluation_stamp_count = stamps.size();
    d.reference_position_bytes = positions.capacity() * sizeof(double);
    d.point_bytes = grid.capacity() * sizeof(grid.front());
    d.provenance_bytes = segments_.capacity() * sizeof(segments_.front()) +
        tiles_.capacity() * sizeof(tiles_.front()) + stamps.capacity() * sizeof(stamps.front());
    d.retained_numeric_bytes = d.reference_position_bytes + d.point_bytes + d.provenance_bytes;
    d.storage_fingerprint = reinterpret_cast<std::uintptr_t>(grid.data()) ^ reinterpret_cast<std::uintptr_t>(positions.data());
    return d;
  }

  std::string provider, schema;
  std::size_t version, ordinal, offset, bases;
  std::vector<double> positions;
  std::vector<wftrain::NonLocalECPReplayPoint> grid;
  std::vector<wftrain::NonLocalECPReplaySegment> segments_;
  std::vector<wftrain::NonLocalECPReplayTile> tiles_;
  std::vector<WaveFunctionComponent::EvaluationStamp> stamps;
};
#endif
} // namespace

TEST_CASE("PsiFormer paired CPU trial isolation preserves the shared source family",
          "[wavefunction][psiformer][complex_wf][complex_parameter_isolation]")
{
  using namespace wftrain;
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell), sibling_p = electrons(simulation_cell);
  auto trial_p = electrons(simulation_cell), reference_p = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  auto initialized = parameters();
  PsiFormerComplexWF component("isolation", initialized, p, source, clone_cap, {.17, -.11, .07});
  const auto initial = component.snapshotParameters();
  component.publishParameters(initial, initial.version);
  REQUIRE(component.snapshotParameters().version > 0);
  auto sibling_base = component.makeClone(sibling_p), trial_base = component.makeClone(trial_p);
  auto& sibling = dynamic_cast<PsiFormerComplexWF&>(*sibling_base);
  auto& trial = dynamic_cast<PsiFormerComplexWF&>(*trial_base);
  // The retained source operator owns its original family, not the later trial family.
  const auto prepare = [&](PsiFormerComplexWF& wf, ParticleSet& particles) {
    RefVectorWithLeader<WaveFunctionComponent> wavefunctions(wf);
    wavefunctions.push_back(wf);
    RefVectorWithLeader<ParticleSet> configurations(particles);
    configurations.push_back(particles);
    return wf.makeBudgetedStreamingDerivativeOperator(wavefunctions, configurations, 0, 0, 5,
                                                     wf.streamingDerivativeRequiredBytes(1));
  };
  const auto score = [&](StreamingDerivativeOperator& stream) {
    const auto& schema = stream.parameterSchema();
    const C one{1};
    const VJPCoefficientChannel channel{"score", DerivativeProduct::SCORE_VJP,
        {schema.providerId(), schema.fingerprint(), stream.parameterVersion(), 0, 0, {&one, 1}}, 0};
    ComplexStreamingSink sink;
    stream.applyVJPs({&channel, 1}, DerivativeAdjoint::TRANSPOSE, sink);
    return sink.values[0];
  };
  const auto original = component.snapshotParameters();
  const auto original_log = evaluate(component, p);
  const auto original_g = p.G;
  const auto original_l = p.L;
  evaluate(sibling, sibling_p);
  evaluate(trial, trial_p);
  trial_p.makeMove(0, ParticleSet::PosType{.005, -.004, .003});
  trial.ratio(trial_p, 0); // Warm the proposal VALUE lane independently of FULL and score.
  CHECK_THROWS(trial.isolateParameterState());
  CHECK_THROWS(component.isolateParameterState());
  trial.restore(0);
  trial_p.rejectMove(0);
  auto retained = prepare(component, p);
  const auto original_score = score(*retained);
  auto sibling_stream = prepare(sibling, sibling_p);
  CHECK(score(*sibling_stream) == original_score);
  trial.isolateParameterState();
  CHECK(trial.snapshotParameters().values == original.values);
  CHECK(trial.snapshotParameters().version == original.version);
  auto candidate = original;
  for (const auto& leaf : initialized.layouts)
    if (leaf.name == "w" || leaf.name == "w_imag")
      candidate.values[leaf.begin] += leaf.name == "w_imag" ? .04 : -.025;
  for (int publication = 0; publication < 2; ++publication)
  {
    candidate.version = trial.snapshotParameters().version;
    candidate.values[0] += .001;
    CHECK(trial.publishParameters(candidate, candidate.version) == candidate.version + 1);
    initialized.values = candidate.values;
    PsiFormerComplexWF reference("independent", initialized, reference_p, source, clone_cap, {.17, -.11, .07});
    const auto actual = evaluate(trial, trial_p);
    close(actual, evaluate(reference, reference_p));
    CHECK(std::abs(std::exp(actual - original_log) - C{1}) > 1e-8);
    for (int electron = 0; electron < p.getTotalNum(); ++electron)
    {
      close(trial_p.L[electron], reference_p.L[electron]);
      for (int dimension = 0; dimension < 3; ++dimension)
        close(trial_p.G[electron][dimension], reference_p.G[electron][dimension]);
    }
    auto trial_stream = prepare(trial, trial_p), reference_stream = prepare(reference, reference_p);
    const auto actual_score = score(*trial_stream), expected_score = score(*reference_stream);
    REQUIRE(actual_score.size() == expected_score.size());
    for (std::size_t i = 0; i < actual_score.size(); ++i)
      close(actual_score[i], expected_score[i]);
    CHECK(component.snapshotParameters().values == original.values);
    CHECK(sibling.snapshotParameters().values == original.values);
    CHECK(component.snapshotParameters().version == original.version);
    CHECK(sibling.snapshotParameters().version == original.version);
    CHECK(score(*retained) == original_score);
    CHECK(evaluate(component, p) == original_log);
    CHECK(p.G == original_g);
    CHECK(p.L == original_l);
    CHECK(evaluate(sibling, sibling_p) == original_log);
    CHECK(sibling_p.G == original_g);
    CHECK(sibling_p.L == original_l);
  }
  const auto accepted = trial.snapshotParameters();
  CHECK_THROWS(trial.publishParameters(candidate, original.version));
  candidate = accepted;
  candidate.values[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(trial.publishParameters(candidate, accepted.version));
  CHECK(trial.snapshotParameters().values == accepted.values);
  CHECK(trial.snapshotParameters().version == accepted.version);
}

TEST_CASE("PsiFormer complex budgeted streams fuse complete score and kinetic contractions",
          "[wavefunction][psiformer][complex_wf][complex_streaming]")
{
  using namespace wftrain;
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell), clone_p = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  const int mass = p.getSpeciesSet().findAttribute("mass");
  p.getSpeciesSet()(mass, 0) = clone_p.getSpeciesSet()(mass, 0) = .7;
  p.getSpeciesSet()(mass, 1) = clone_p.getSpeciesSet()(mass, 1) = 1.9;
  p.resetGroups();
  clone_p.resetGroups();
  clone_p.R[1][2] += .037;
  clone_p.update();
  PsiFormerComplexWF component("stream", parameters(), p, source, clone_cap, {.17, -.11, .07});
  auto base_clone = component.makeClone(clone_p);
  auto& clone = dynamic_cast<PsiFormerComplexWF&>(*base_clone);
  RefVectorWithLeader<WaveFunctionComponent> components(component);
  components.push_back(component);
  components.push_back(clone);
  RefVectorWithLeader<ParticleSet> particles(p);
  particles.push_back(p);
  particles.push_back(clone_p);
  const auto bytes = component.streamingDerivativeRequiredBytes(2);
  const auto lane_bytes = component.streamingDerivativeRequiredBytes(0);
  CHECK(bytes == lane_bytes + 4 * 2 * 4 * sizeof(double) + 2 * sizeof(C));
  for (const auto cap : {std::size_t{0}, bytes - 1, std::numeric_limits<std::size_t>::max()})
    CHECK_THROWS(component.makeBudgetedStreamingDerivativeOperator(components, particles, 7, 11, 5, cap));
  CHECK_THROWS(component.makeStreamingDerivativeOperator(components, particles, 7, 11, 5));
  auto stream = component.makeBudgetedStreamingDerivativeOperator(components, particles, 7, 11, 5, bytes);
  const auto diagnostic = stream->storageDiagnostics();
  CHECK(diagnostic.retained_numeric_bytes == bytes);
  CHECK(diagnostic.shared_workspace_bytes == lane_bytes);
  CHECK(diagnostic.real_parameter_vectors == 2);
  CHECK(diagnostic.complex_parameter_vectors == 5);
  CHECK(stream->capabilities().maximum_vjp_channels == 3);
  CHECK(stream->capabilities().supports(DerivativeProduct::SCORE_JVP));
  CHECK(stream->capabilities().supports(DerivativeAdjoint::HERMITIAN));
  CHECK(diagnostic.sample_product_bytes == 2 * sizeof(C));
  CHECK(component.structuredParameterProvider() == &component);
  const auto& schema = stream->parameterSchema();
  std::array<std::vector<C>, 3> coefficients{{{{.3, -.2}, {-.4, .1}},
      {{.7, .15}, {.2, -.35}}, {{-.6, .4}, {.8, -.1}}}};
  std::array<VJPCoefficientChannel, 3> channels;
  for (std::size_t c = 0; c < 3; ++c)
  {
    channels[c] = {c == 0 ? "score0" : c == 1 ? "kinetic" : "score2",
        c == 1 ? DerivativeProduct::LOCAL_ENERGY_VJP : DerivativeProduct::SCORE_VJP,
        {schema.providerId(), schema.fingerprint(), stream->parameterVersion(), 7, 11,
          {coefficients[c].data(), 2}}, c == 1 ? localEnergyTermBit(LocalEnergyTerm::KINETIC) : 0};
  }
  // Explicit complete rows are test-only; the fused owner never materializes these B-by-P matrices.
  std::array<std::vector<C>, 2> score, kinetic;
  const C one{1};
  for (std::size_t sample = 0; sample < 2; ++sample)
  {
    RefVectorWithLeader<WaveFunctionComponent> one_component(component);
    one_component.push_back(components[sample]);
    RefVectorWithLeader<ParticleSet> one_particle(p);
    one_particle.push_back(particles[sample]);
    auto oracle = component.makeBudgetedStreamingDerivativeOperator(one_component, one_particle, sample, 0, 7,
        component.streamingDerivativeRequiredBytes(1));
    CHECK(oracle->storageDiagnostics().shared_workspace_identity == diagnostic.shared_workspace_identity);
    std::array<VJPCoefficientChannel, 2> probes{{
        {"score", DerivativeProduct::SCORE_VJP,
          {schema.providerId(), schema.fingerprint(), oracle->parameterVersion(), sample, 0, {&one, 1}}, 0},
        {"kinetic", DerivativeProduct::LOCAL_ENERGY_VJP,
          {schema.providerId(), schema.fingerprint(), oracle->parameterVersion(), sample, 0, {&one, 1}},
          localEnergyTermBit(LocalEnergyTerm::KINETIC)}}};
    ComplexStreamingSink sink;
    oracle->applyVJPs({probes.data(), 2}, DerivativeAdjoint::TRANSPOSE, sink);
    score[sample] = sink.values[0];
    kinetic[sample] = sink.values[1];
  }
  // Mutable ParticleSets cannot change an already prepared sample snapshot.
  p.R[0][0] += .019;
  p.update();
  ComplexStreamingSink sink;
  stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, sink);
  for (std::size_t c = 0; c < 3; ++c)
    for (std::size_t parameter = 0; parameter < schema.parameterCount(); ++parameter)
      close(sink.values[c][parameter], coefficients[c][0] * (c == 1 ? kinetic[0] : score[0])[parameter] +
          coefficients[c][1] * (c == 1 ? kinetic[1] : score[1])[parameter], 2e-11);
  sink.reset();
  sink.fail_next = true;
  CHECK_THROWS(stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, sink));
  CHECK(sink.state() == DerivativeSinkState::POISONED);
  sink.reset();
  CHECK_NOTHROW(stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, sink));
  CHECK(stream->storageDiagnostics().retained_numeric_bytes == diagnostic.retained_numeric_bytes);
  CHECK(stream->storageDiagnostics().storage_fingerprint == diagnostic.storage_fingerprint);
  auto unsupported = channels;
  unsupported[1].local_energy_term_mask |= localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);
  sink.reset();
  CHECK_THROWS(stream->applyVJPs({unsupported.data(), 3}, DerivativeAdjoint::TRANSPOSE, sink));
  CHECK(sink.state() == DerivativeSinkState::IDLE);
  const auto snapshot = component.snapshotParameters();
  component.publishParameters(snapshot, snapshot.version);
  CHECK_THROWS(stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, sink));
  CHECK(sink.state() == DerivativeSinkState::POISONED);
}

TEST_CASE("PsiFormer complex stream physical finite differences and concrete factory rejection",
          "[wavefunction][psiformer][complex_wf][complex_streaming]")
{
  using namespace wftrain;
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell), clone_p = electrons(simulation_cell), foreign_p = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  const int mass = p.getSpeciesSet().findAttribute("mass");
  p.getSpeciesSet()(mass, 0) = clone_p.getSpeciesSet()(mass, 0) = .8;
  p.getSpeciesSet()(mass, 1) = clone_p.getSpeciesSet()(mass, 1) = 1.7;
  p.resetGroups();
  clone_p.resetGroups();
  PsiFormerComplexWF component("fdstream", parameters(), p, source, clone_cap, {.13, -.09, .04});
  auto base_clone = component.makeClone(clone_p);
  auto& clone = dynamic_cast<PsiFormerComplexWF&>(*base_clone);
  PsiFormerComplexWF foreign("foreign", parameters(), foreign_p, source, clone_cap);
  RefVectorWithLeader<WaveFunctionComponent> components(component);
  components.push_back(component);
  RefVectorWithLeader<ParticleSet> particles(p);
  particles.push_back(p);
  const auto cap = component.streamingDerivativeRequiredBytes(1);
  const auto prepare = [&] {
    return component.makeBudgetedStreamingDerivativeOperator(components, particles, 0, 0, 3, cap);
  };
  const auto original = component.snapshotParameters();
  const auto log_center = evaluate(component, p);
  std::vector<C> score, kinetic;
  const double epsilon = 2e-5;
  // One independent public-component FD per canonical role, including imaginary heads and cusp.
  for (const auto& block : component.parameterSchema().blocks())
  {
    std::array<C, 2> log, energy;
    for (std::size_t sign = 0; sign < 2; ++sign)
    {
      auto candidate = component.snapshotParameters();
      candidate.values = original.values;
      candidate.values[block.offset] += sign == 0 ? epsilon : -epsilon;
      component.publishParameters(candidate, candidate.version);
      log[sign] = evaluate(component, p);
      energy[sign] = component.observeCoulombEnergy(p, 0).kinetic;
    }
    score.push_back((std::exp(log[0] - log_center) - std::exp(log[1] - log_center)) / (2 * epsilon));
    kinetic.push_back((energy[0] - energy[1]) / (2 * epsilon));
  }
  auto restored = component.snapshotParameters();
  restored.values = original.values;
  component.publishParameters(restored, restored.version);
  auto stream = prepare();
  const auto& schema = component.parameterSchema();
  const C one{1};
  std::array<VJPCoefficientChannel, 2> channels{{
      {"score", DerivativeProduct::SCORE_VJP,
        {schema.providerId(), schema.fingerprint(), stream->parameterVersion(), 0, 0, {&one, 1}}, 0},
      {"kinetic", DerivativeProduct::LOCAL_ENERGY_VJP,
        {schema.providerId(), schema.fingerprint(), stream->parameterVersion(), 0, 0, {&one, 1}},
        localEnergyTermBit(LocalEnergyTerm::KINETIC)}}};
  ComplexStreamingSink sink;
  stream->applyVJPs({channels.data(), 2}, DerivativeAdjoint::TRANSPOSE, sink);
  for (std::size_t block = 0; block < schema.blocks().size(); ++block)
  {
    close(sink.values[0][schema.blocks()[block].offset], score[block], 3e-6);
    close(sink.values[1][schema.blocks()[block].offset], kinetic[block], 3e-5);
  }
  RefVectorWithLeader<WaveFunctionComponent> wrong(component);
  wrong.push_back(foreign);
  CHECK_THROWS(component.makeBudgetedStreamingDerivativeOperator(wrong, particles, 0, 0, 3, cap));
  RefVectorWithLeader<ParticleSet> wrong_particles(p);
  wrong_particles.push_back(clone_p);
  CHECK_THROWS(component.makeBudgetedStreamingDerivativeOperator(components, wrong_particles, 0, 0, 3, cap));
  p.getSpeciesSet()(mass, 0) = 0;
  p.resetGroups();
  CHECK_THROWS(prepare());
  p.getSpeciesSet()(mass, 0) = .8;
  p.resetGroups();
  CHECK_NOTHROW(prepare());
  evaluate(clone, clone_p);
  clone_p.makeMove(0, ParticleSet::PosType{.007, -.011, .013});
  clone.ratio(clone_p, 0);
  CHECK_THROWS(prepare()); // another clone's unresolved proposal blocks the whole family
  sink.reset();
  CHECK_THROWS(stream->applyVJPs({channels.data(), 2}, DerivativeAdjoint::TRANSPOSE, sink));
  clone.restore(0);
  clone_p.rejectMove(0);
  sink.reset();
  CHECK_NOTHROW(stream->applyVJPs({channels.data(), 2}, DerivativeAdjoint::TRANSPOSE, sink));
  CHECK(component.structuredParameterProvider() == &component);
}

TEST_CASE("PsiFormer complex real-tangent score products preserve phase and both adjoints",
          "[wavefunction][psiformer][complex_wf][complex_score_products]")
{
  using namespace wftrain;
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell), clone_p = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  clone_p.R[1][2] += .037;
  clone_p.update();
  PsiFormerComplexWF component("products", parameters(), p, source, clone_cap, {.17, -.11, .07});
  auto base_clone = component.makeClone(clone_p);
  auto& clone = dynamic_cast<PsiFormerComplexWF&>(*base_clone);
  const auto original = component.snapshotParameters();
  const auto& schema = component.parameterSchema();
  std::vector<C> direction(schema.parameterCount());
  std::size_t block_index = 0;
  for (const auto& block : schema.blocks())
    direction[block.offset] = (++block_index % 2) ? .08 : -.05;
  const std::array<C, 2> reference{evaluate(component, p), evaluate(clone, clone_p)};
  std::array<std::array<C, 2>, 2> perturbed;
  const double epsilon = 2e-5;
  for (std::size_t sign = 0; sign < 2; ++sign)
  {
    auto candidate = component.snapshotParameters();
    candidate.values = original.values;
    for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
      candidate.values[parameter] += (sign == 0 ? epsilon : -epsilon) * direction[parameter].real();
    component.publishParameters(candidate, candidate.version);
    perturbed[sign] = {evaluate(component, p), evaluate(clone, clone_p)};
  }
  auto restored = component.snapshotParameters();
  restored.values = original.values;
  component.publishParameters(restored, restored.version);
  RefVectorWithLeader<WaveFunctionComponent> components(component);
  components.push_back(component);
  components.push_back(clone);
  RefVectorWithLeader<ParticleSet> particles(p);
  particles.push_back(p);
  particles.push_back(clone_p);
  const auto bytes = component.streamingDerivativeRequiredBytes(2);
  auto stream = component.makeBudgetedStreamingDerivativeOperator(components, particles, 3, 7, 5, bytes);
  const auto storage = stream->storageDiagnostics();
  const StructuredParameterVectorConstView tangent(schema, stream->parameterVersion(),
      {direction.data(), direction.size()});
  ComplexSampleSink samples;
  stream->applyScoreJVP(tangent, samples);
  for (std::size_t sample = 0; sample < 2; ++sample)
    close(samples.values[sample], (std::exp(perturbed[0][sample] - reference[sample]) -
        std::exp(perturbed[1][sample] - reference[sample])) / (2 * epsilon), 3e-6);

  // Compare the actual neural streaming covariance on a two-direction REAL tangent subspace.
  std::vector<C> phase_direction(direction.size());
  for (const auto& block : schema.blocks())
    if (block.id.find("/w_imag") != std::string::npos)
      phase_direction[block.offset] = .6;
  ComplexSampleSink phase_products;
  stream->applyScoreJVP({schema, stream->parameterVersion(), {phase_direction.data(), phase_direction.size()}},
                        phase_products);
  const std::array<double, 2> weights{.4, 1.3};
  const StochasticReconfigurationBatch batch{stream.get(), {weights.data(), weights.size()}};
  StochasticReconfigurationOperator covariance(schema, stream->parameterVersion(), {&batch, 1});
  CHECK(covariance.storageDiagnostics().retained_numeric_bytes ==
        2 * direction.size() * sizeof(C) + 2 * (sizeof(double) + sizeof(C)));
  const std::array<const std::vector<C>*, 2> directions{&direction, &phase_direction};
  const std::array<const std::vector<C>*, 2> products{&samples.values, &phase_products.values};
  std::array<C, 2> means{};
  for (std::size_t row = 0; row < 2; ++row)
    for (std::size_t sample = 0; sample < 2; ++sample)
      means[row] += weights[sample] * (*products[row])[sample] / 1.7;
  for (std::size_t column = 0; column < 2; ++column)
  {
    std::vector<C> action(direction.size());
    covariance.apply({schema, stream->parameterVersion(), {directions[column]->data(), direction.size()}},
                     {action.data(), action.size()});
    for (std::size_t row = 0; row < 2; ++row)
    {
      double dense = 0, actual = 0;
      for (std::size_t sample = 0; sample < 2; ++sample)
        dense += weights[sample] * std::real(std::conj((*products[row])[sample] - means[row]) *
            ((*products[column])[sample] - means[column])) / 1.7;
      for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
        actual += (*directions[row])[parameter].real() * action[parameter].real();
      CHECK(actual == Approx(dense).margin(2e-10));
    }
  }

  std::array<C, 2> coefficients{{{.3, -.2}, {-.4, .1}}};
  std::array<VJPCoefficientChannel, 2> channels{{
      {"score", DerivativeProduct::SCORE_VJP,
        {schema.providerId(), schema.fingerprint(), stream->parameterVersion(), 3, 7, {coefficients.data(), 2}}, 0},
      {"kinetic", DerivativeProduct::LOCAL_ENERGY_VJP,
        {schema.providerId(), schema.fingerprint(), stream->parameterVersion(), 3, 7, {coefficients.data(), 2}},
        localEnergyTermBit(LocalEnergyTerm::KINETIC)}}};
  ComplexStreamingSink transpose, hermitian, conjugated;
  stream->applyVJPs({channels.data(), 2}, DerivativeAdjoint::TRANSPOSE, transpose);
  stream->applyVJPs({channels.data(), 2}, DerivativeAdjoint::HERMITIAN, hermitian);
  C transpose_dot{}, hermitian_dot{}, transpose_sample_dot{}, hermitian_sample_dot{};
  for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
  {
    transpose_dot += direction[parameter].real() * transpose.values[0][parameter];
    hermitian_dot += direction[parameter].real() * hermitian.values[0][parameter];
  }
  for (std::size_t sample = 0; sample < 2; ++sample)
  {
    transpose_sample_dot += coefficients[sample] * samples.values[sample];
    hermitian_sample_dot += coefficients[sample] * std::conj(samples.values[sample]);
  }
  close(transpose_dot, transpose_sample_dot);
  close(hermitian_dot, hermitian_sample_dot);
  for (auto& value : coefficients)
    value = std::conj(value);
  stream->applyVJPs({channels.data(), 2}, DerivativeAdjoint::TRANSPOSE, conjugated);
  // Global HERMITIAN capability also covers complete kinetic responses, not only the score metric.
  for (std::size_t channel = 0; channel < 2; ++channel)
    for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
      close(hermitian.values[channel][parameter], std::conj(conjugated.values[channel][parameter]));
  CHECK(stream->storageDiagnostics().retained_numeric_bytes == bytes);
  CHECK(stream->storageDiagnostics().storage_fingerprint == storage.storage_fingerprint);
  samples.reset();
  const C first = direction[0];
  direction[0] = {0, 1};
  CHECK_THROWS(stream->applyScoreJVP(tangent, samples));
  CHECK(samples.state() == DerivativeSinkState::IDLE);
  direction[0] = first;
  samples.fail_next = true;
  CHECK_THROWS(stream->applyScoreJVP(tangent, samples));
  CHECK(samples.state() == DerivativeSinkState::POISONED);
  samples.reset();
  CHECK_NOTHROW(stream->applyScoreJVP(tangent, samples));
  samples.reset();
  bool published = false;
  samples.callback = [&] {
    if (!published)
    {
      // Taking an exclusive publication lease here proves callbacks are outside the model read lock.
      const auto snapshot = component.snapshotParameters();
      component.publishParameters(snapshot, snapshot.version);
      published = true;
    }
  };
  CHECK_NOTHROW(stream->applyScoreJVP(tangent, samples));
  CHECK(published);
  samples.callback = {};
  samples.reset();
  CHECK_THROWS(stream->applyScoreJVP(tangent, samples));
  CHECK(samples.state() == DerivativeSinkState::POISONED);
  auto fresh = component.makeBudgetedStreamingDerivativeOperator(components, particles, 3, 7, 5, bytes);
  const StructuredParameterVectorConstView fresh_tangent(schema, fresh->parameterVersion(),
      {direction.data(), direction.size()});
  evaluate(clone, clone_p);
  clone_p.makeMove(0, ParticleSet::PosType{.007, -.011, .013});
  clone.ratio(clone_p, 0);
  samples.reset();
  CHECK_THROWS(fresh->applyScoreJVP(fresh_tangent, samples));
  clone.restore(0);
  clone_p.rejectMove(0);
  samples.reset();
  CHECK_NOTHROW(fresh->applyScoreJVP(fresh_tangent, samples));
}

#ifdef ENABLE_CUDA
TEST_CASE("PsiFormer complex GPU inference shares one capped lane without CPU tapes",
          "[wavefunction][psiformer][complex_gpu_inference]")
{
  namespace dev = ps::device;
  int devices = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices)
  {
    WARN("NOT_RUN: complex GPU inference requires an available HIP device");
    return;
  }
  REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
  {
    WARN("NOT_RUN: complex GPU inference requires an available CUDA device");
    return;
  }
  REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
  auto& metrics = dev::deviceMemoryMetrics(ordinal);
  const auto lattice = cell();
  const SimulationCell simulation_cell(lattice);
  for (const auto twist : {std::array<double, 3>{}, std::array<double, 3>{.17, -.11, .07}})
  {
    auto cpu_p = electrons(simulation_cell), gpu_p = electrons(simulation_cell), clone_p = electrons(simulation_cell);
    auto source = ions(simulation_cell);
    const auto initialized = parameters();
    ps::ExecutionEnvironment env;
    env.boundary = ps::BoundaryCondition::PERIODIC;
    env.geometry_feature_policy = ps::GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    env.periodic_axes = {true, true, true};
    env.head_representation = ps::HeadRepresentation::PAIRED_REAL_COMPLEX;
    env.amplitude_scalar_domain = ps::ScalarDomain::COMPLEX;
    for (int a = 0; a < 3; ++a)
      for (int d = 0; d < 3; ++d)
        env.lattice_vectors[a][d] = lattice.R(a, d);
    const auto plan = ps::PsiFormerExecutionPlan::fromParameters(initialized, initialized.model_shape, env);
    const auto exact = dev::estimateComplexValueCost(plan, 1).device_bytes +
                       dev::estimateComplexSpatialCost(plan, 1).device_bytes;
    const auto before_reject = metrics.snapshot();
    CHECK_THROWS_AS(PsiFormerComplexWF("short", initialized, gpu_p, source, clone_cap, twist, exact - 1), std::length_error);
    CHECK_THROWS_AS(PsiFormerComplexWF("shortGL", initialized, gpu_p, source, 1, twist, exact), std::length_error);
    CHECK_THROWS_AS(PsiFormerComplexWF("unlimited", initialized, gpu_p, source, clone_cap, twist,
        std::numeric_limits<std::size_t>::max()), std::invalid_argument);
    CHECK(metrics.snapshot().allocation_calls == before_reject.allocation_calls);
    PsiFormerComplexWF cpu("cpu", initialized, cpu_p, source, clone_cap, twist);
    const auto before = metrics.snapshot();
    PsiFormerComplexWF gpu("gpu", initialized, gpu_p, source, clone_cap, twist, exact);
    REQUIRE(gpu.gpuInferenceEnabled());
    const auto storage = gpu.numericStorage();
    CHECK(storage.shared_gpu_bytes == exact);
    CHECK(metrics.snapshot().live_bytes - before.live_bytes == exact + storage.shared_image_bytes);
    CHECK(storage.allocated_bytes == 4 * (sizeof(WaveFunctionComponent::GradType) + sizeof(WaveFunctionComponent::ValueType)));
    CHECK(storage.required_bytes == storage.allocated_bytes); // no CPU VALUE/FULL/ACTIVE tapes
    const auto cold_clone = metrics.snapshot();
    auto base = gpu.makeClone(clone_p);
    auto& clone = dynamic_cast<PsiFormerComplexWF&>(*base);
    CHECK(metrics.snapshot().allocation_calls == cold_clone.allocation_calls);
    CHECK(clone.numericStorage().shared_image_identity == storage.shared_image_identity);
    CHECK(clone.numericStorage().allocated_bytes == storage.allocated_bytes);
    clone_p.makeMove(0, ParticleSet::PosType{.01, 0, 0});
    CHECK_THROWS(clone.ratio(clone_p, 0)); // cold clones never inherit accepted scalar state
    clone_p.rejectMove(0);
    const auto warm = metrics.snapshot();
    for (int repeat = 0; repeat < 2; ++repeat)
    {
      close(evaluate(gpu, gpu_p), evaluate(cpu, cpu_p));
      close(evaluate(clone, clone_p), cpu.get_log_value());
      for (int e = 0; e < 4; ++e)
      {
        for (int d = 0; d < 3; ++d)
          close(gpu_p.G[e][d], cpu_p.G[e][d]);
        close(gpu_p.L[e], cpu_p.L[e]);
      }
      for (int e : {0, 2})
      {
        const auto accepted = cpu.evalGrad(cpu_p, e), actual = gpu.evalGrad(gpu_p, e);
        for (int d = 0; d < 3; ++d)
          close(actual[d], accepted[d]);
        const ParticleSet::PosType delta{.013, -.017, .011};
        cpu_p.makeMove(e, delta);
        gpu_p.makeMove(e, delta);
        WaveFunctionComponent::GradType cpu_g{}, gpu_g{};
        close(gpu.ratio(gpu_p, e), cpu.ratio(cpu_p, e));
        close(gpu.ratioGrad(gpu_p, e, gpu_g), cpu.ratioGrad(cpu_p, e, cpu_g));
        for (int d = 0; d < 3; ++d)
          close(gpu_g[d], cpu_g[d]);
        if (repeat)
        {
          gpu.acceptMove(gpu_p, e); cpu.acceptMove(cpu_p, e);
          gpu_p.acceptMove(e); cpu_p.acceptMove(e);
        }
        else
        {
          gpu.restore(e); cpu.restore(e);
          gpu_p.rejectMove(e); cpu_p.rejectMove(e);
        }
      }
    }
    CHECK(metrics.snapshot().allocation_calls == warm.allocation_calls);
    CHECK(metrics.snapshot().free_calls == warm.free_calls);
    CHECK(gpu.numericStorage().fingerprint == storage.fingerprint);
    const auto snapshot = gpu.snapshotParameters();
    CHECK_THROWS_AS(gpu.publishParameters(snapshot, snapshot.version), std::logic_error);
    CHECK(gpu.snapshotParameters().values == snapshot.values);
    CHECK(gpu.structuredParameterProvider() == nullptr);
    CHECK_THROWS_AS(gpu.streamingDerivativeRequiredBytes(1), std::logic_error);
    RefVectorWithLeader<WaveFunctionComponent> components(gpu);
    components.push_back(gpu);
    RefVectorWithLeader<ParticleSet> particles(gpu_p);
    particles.push_back(gpu_p);
    CHECK_THROWS_AS(gpu.makeBudgetedStreamingDerivativeOperator(components, particles, 0, 0, 1, clone_cap),
                    std::logic_error);
    CHECK(gpu.numericStorage().fingerprint == storage.fingerprint);
    WaveFunctionComponent::WFBufferType buffer;
    gpu.registerData(gpu_p, buffer);
    buffer.allocate();
    buffer.rewind();
    gpu.updateBuffer(gpu_p, buffer, true);
    const auto log = gpu.get_log_value();
    buffer.rewind();
    gpu.copyFromBuffer(gpu_p, buffer);
    close(gpu.get_log_value(), log);
    const auto original = gpu_p.R[0];
    gpu_p.R[0] = source.R[0];
    gpu_p.update();
    CHECK_THROWS(evaluate(gpu, gpu_p)); // unavailable FULL gradients abort; no CPU fallback
    gpu_p.R[0] = original;
    gpu_p.update();
    CHECK_NOTHROW(evaluate(gpu, gpu_p));
  }
}

TEST_CASE("PsiFormer explicit paired GPU target lane preserves admission and clone ownership",
          "[wavefunction][psiformer][complex_gpu_target_inference]")
{
  namespace dev = ps::device;
  int devices = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices)
  {
    WARN("NOT_RUN: paired GPU target binding requires an available HIP device");
    return;
  }
  REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
  {
    WARN("NOT_RUN: paired GPU target binding requires an available CUDA device");
    return;
  }
  REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
  ScopedComplexGpuControls controls;
  auto& metrics = dev::deviceMemoryMetrics(ordinal);
  const auto lattice = cell();
  const SimulationCell simulation_cell(lattice);
  for (const auto twist : {std::array<double, 3>{}, std::array<double, 3>{.17, -.11, .07}})
  {
    auto cpu_p = electrons(simulation_cell), gpu_p = electrons(simulation_cell), clone_p = electrons(simulation_cell);
    auto source = ions(simulation_cell);
    const auto initialized = parameters();
    ps::ExecutionEnvironment env;
    env.boundary = ps::BoundaryCondition::PERIODIC;
    env.geometry_feature_policy = ps::GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    env.periodic_axes = {true, true, true};
    env.head_representation = ps::HeadRepresentation::PAIRED_REAL_COMPLEX;
    env.amplitude_scalar_domain = ps::ScalarDomain::COMPLEX;
    for (int a = 0; a < 3; ++a)
      for (int d = 0; d < 3; ++d)
        env.lattice_vectors[a][d] = lattice.R(a, d);
    const auto plan = ps::PsiFormerExecutionPlan::fromParameters(initialized, initialized.model_shape, env);
    const auto value_cost = dev::estimateComplexValueCost(plan, 1);
    const auto spatial_cost = dev::estimateComplexSpatialTargetCost(plan, 3);
    const auto work = dev::estimateHeadFreeTargetPrefixCost(plan, 3).charged_scalar_work_units;
    const auto exact = value_cost.device_bytes + spatial_cost.device_bytes;

    // Controls unsupported by this forward-only lane fail before image/owner allocation; CPU is isolated.
    for (const char* name : {"PSIFORMER_GPU_KINETIC_TARGETS", "PSIFORMER_GPU_LIFTED_CHANNELS",
        "PSIFORMER_GPU_RETAINED_COORDINATE_BYTES", "PSIFORMER_GPU_METROPOLIS_BYTES",
        "PSIFORMER_GPU_METROPOLIS_DRIFT_BYTES", "PSIFORMER_GPU_ADAM_BYTES", "PSIFORMER_GPU_RESIDENT_ENERGY_ADAM_BYTES",
        "PSIFORMER_GPU_RESIDENT_ENERGY_SR_BYTES", "PSIFORMER_GPU_RESIDENT_ENERGY_SR_HOST_BYTES", "PSIFORMER_GPU_RESIDENT_ENERGY_SR_REUSE"})
    {
      controls.set(name, "1");
      CHECK_NOTHROW(PsiFormerComplexWF("cpuIsolated", initialized, cpu_p, source, clone_cap, twist));
      const auto before = metrics.snapshot();
      CHECK_THROWS_AS(PsiFormerComplexWF("unsupported", initialized, gpu_p, source, clone_cap, twist, exact), std::invalid_argument);
      CHECK(metrics.snapshot().allocation_calls == before.allocation_calls);
      ::unsetenv(name);
    }
    controls.set("PSIFORMER_GPU_TARGET_SELECTION", "vgl");
    CHECK_NOTHROW(PsiFormerComplexWF("cpuIsolated", initialized, cpu_p, source, clone_cap, twist));
    const auto before_reject = metrics.snapshot();
    CHECK_THROWS_AS(PsiFormerComplexWF("selection", initialized, gpu_p, source, clone_cap, twist, exact), std::invalid_argument);
    CHECK(metrics.snapshot().allocation_calls == before_reject.allocation_calls);
    controls.set("PSIFORMER_GPU_TARGET_SELECTION", "off");
    controls.target(3, spatial_cost.device_bytes, work);
    CHECK_THROWS_AS(PsiFormerComplexWF("sumCap", initialized, gpu_p, source, clone_cap, twist, exact-1), std::length_error);
    controls.target(3, spatial_cost.device_bytes-1, work);
    CHECK_THROWS_AS(PsiFormerComplexWF("ownerCap", initialized, gpu_p, source, clone_cap, twist, exact), std::length_error);
    controls.target(3, spatial_cost.device_bytes, work-1);
    CHECK_THROWS_AS(PsiFormerComplexWF("workCap", initialized, gpu_p, source, clone_cap, twist, exact), std::length_error);
    controls.target(3, spatial_cost.device_bytes, work);
    for (const auto* spelling : {"0", "", "3x"})
    {
      controls.set("PSIFORMER_GPU_VGL_TARGETS", spelling);
      CHECK_THROWS_AS(PsiFormerComplexWF("malformed", initialized, gpu_p, source, clone_cap, twist, exact), std::invalid_argument);
    }
    controls.target(3, spatial_cost.device_bytes, work);
    controls.set("PSIFORMER_GPU_TARGET_WORKSPACE_BYTES", std::to_string(std::numeric_limits<std::size_t>::max()));
    CHECK_THROWS_AS(PsiFormerComplexWF("unbounded", initialized, gpu_p, source, clone_cap, twist, exact), std::invalid_argument);
    CHECK(metrics.snapshot().allocation_calls == before_reject.allocation_calls);

    controls.target(3, spatial_cost.device_bytes, work);
    PsiFormerComplexWF cpu("cpu", initialized, cpu_p, source, clone_cap, twist);
    const auto before = metrics.snapshot();
    PsiFormerComplexWF gpu("gpu", initialized, gpu_p, source, clone_cap, twist, exact);
    const auto storage = gpu.numericStorage();
    CHECK(storage.shared_gpu_bytes == exact);
    CHECK(storage.shared_gpu_host_bytes == value_cost.host_bytes + spatial_cost.host_bytes +
          sizeof(dev::ComplexValueResult) + 4*(3*sizeof(double)+5*sizeof(cuDoubleComplex)+4*sizeof(C)));
    CHECK(metrics.snapshot().live_bytes-before.live_bytes == exact+storage.shared_image_bytes);
    CHECK(storage.allocated_bytes == 4*(sizeof(WaveFunctionComponent::GradType)+sizeof(WaveFunctionComponent::ValueType)));
    const auto before_clone = metrics.snapshot();
    auto clone = gpu.makeClone(clone_p);
    auto& cloned = dynamic_cast<PsiFormerComplexWF&>(*clone);
    CHECK(cloned.numericStorage().shared_image_identity == storage.shared_image_identity);
    CHECK(metrics.snapshot().allocation_calls == before_clone.allocation_calls);
    const auto warm = metrics.snapshot();
    for (int repeat = 0; repeat < 2; ++repeat)
    {
      close(evaluate(gpu, gpu_p), evaluate(cpu, cpu_p));
      close(evaluate(cloned, clone_p), cpu.get_log_value());
      for (int e = 0; e < 4; ++e)
      {
        for (int d = 0; d < 3; ++d)
          close(gpu_p.G[e][d], cpu_p.G[e][d]);
        close(gpu_p.L[e], cpu_p.L[e]);
      }
      cpu_p.makeMove(3, ParticleSet::PosType{.013, -.017, .011});
      gpu_p.makeMove(3, ParticleSet::PosType{.013, -.017, .011});
      WaveFunctionComponent::GradType cpu_g{}, gpu_g{};
      close(gpu.ratioGrad(gpu_p, 3, gpu_g), cpu.ratioGrad(cpu_p, 3, cpu_g));
      for (int d = 0; d < 3; ++d)
        close(gpu_g[d], cpu_g[d]);
      gpu.restore(3); cpu.restore(3);
      gpu_p.rejectMove(3); cpu_p.rejectMove(3);
    }
    CHECK(metrics.snapshot().allocation_calls == warm.allocation_calls);
    CHECK(metrics.snapshot().free_calls == warm.free_calls);
    CHECK(gpu.numericStorage().fingerprint == storage.fingerprint);
  }
}
#endif

TEST_CASE("PsiFormer paired parameter warmstart precedes identity and immutable GPU image",
          "[wavefunction][psiformer][complex_wf][complex_parameter_warmstart]")
{
  const SimulationCell simulation_cell(cell());
  auto original_p = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  const auto initialized = parameters();
  PsiFormerComplexWF original("warm", initialized, original_p, source, clone_cap);
  auto candidate = original.snapshotParameters();
  candidate.values[0] += .015;
  for (const auto& block : original.parameterSchema().blocks())
    if (block.id.find("/w_imag") != std::string::npos)
      candidate.values[block.offset] -= .012;
  const auto version = original.publishParameters(candidate, candidate.version);
  const auto expected = original.snapshotParameters();
  const std::filesystem::path root = "psiformer_paired_parameter_warmstart.root.h5";
  // Delete only this test-owned native root, including on an assertion exception.
  struct Cleanup
  {
    std::filesystem::path path;
    ~Cleanup()
    {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  } cleanup{root};
  wftrain::TrainingIterationState progress{2, version, original.parameterSchema().fingerprint()};
  wftrain::TrainingStageState stage{"old-training-stage", 0, 2, "old-physics-not-restored"};
  wftrain::TrainingCheckpoint::saveAtomic(root, original, progress, stage);
  const auto read_bytes = [&] {
    std::ifstream input(root, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  };
  const auto source_bytes = read_bytes();
  REQUIRE_FALSE(source_bytes.empty());
  for (const auto twist : {std::array<double, 3>{}, std::array<double, 3>{.17, -.11, .07}})
  {
    auto cpu_p = electrons(simulation_cell), reference_p = electrons(simulation_cell);
    auto changed = initialized;
    changed.values = expected.values;
    PsiFormerComplexWF reference("warm", changed, reference_p, source, clone_cap, twist);
    PsiFormerComplexWF cpu("warm", initialized, cpu_p, source, clone_cap, twist, 0, root.string());
    CHECK(cpu.snapshotParameters().values == expected.values);
    close(evaluate(cpu, cpu_p), evaluate(reference, reference_p));
    for (int e = 0; e < 4; ++e)
    {
      for (int d = 0; d < 3; ++d)
        close(cpu_p.G[e][d], reference_p.G[e][d]);
      close(cpu_p.L[e], reference_p.L[e]);
    }
#ifdef ENABLE_CUDA
    const std::size_t gpu_cap = 64 * 1024 * 1024;
    int devices = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
    const bool available = hipGetDeviceCount(&devices) == hipSuccess && devices > 0;
    if (available)
      REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
    const bool available = cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
    if (available)
      REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
    if (!available)
    {
      WARN("NOT_RUN: warmstart GPU parity requires an available device");
      continue;
    }
    ScopedComplexGpuControls controls;
    auto& metrics = ps::device::deviceMemoryMetrics(ordinal);
    auto gpu_p = electrons(simulation_cell), clone_p = electrons(simulation_cell);
    const auto before_reject = metrics.snapshot();
    CHECK_THROWS_AS(PsiFormerComplexWF("foreign", initialized, gpu_p, source, clone_cap,
                                     twist, gpu_cap, root.string()), std::runtime_error);
    CHECK(metrics.snapshot().allocation_calls == before_reject.allocation_calls);
    PsiFormerComplexWF gpu("warm", initialized, gpu_p, source, clone_cap, twist, gpu_cap, root.string());
    CHECK(gpu.snapshotParameters().values == expected.values);
    CHECK(gpu.structuredParameterProvider() == nullptr);
    auto base = gpu.makeClone(clone_p);
    auto& clone = dynamic_cast<PsiFormerComplexWF&>(*base);
    CHECK(clone.numericStorage().shared_image_identity == gpu.numericStorage().shared_image_identity);
    CHECK(gpu.numericStorage().allocated_bytes == cpu_p.G.size() * sizeof(WaveFunctionComponent::GradType) +
          cpu_p.L.size() * sizeof(WaveFunctionComponent::ValueType));
    close(evaluate(gpu, gpu_p), cpu.get_log_value());
    close(evaluate(clone, clone_p), cpu.get_log_value());
    for (int e = 0; e < 4; ++e)
    {
      for (int d = 0; d < 3; ++d)
        close(gpu_p.G[e][d], cpu_p.G[e][d]);
      close(gpu_p.L[e], cpu_p.L[e]);
    }
    CHECK_THROWS(gpu.publishParameters(expected, expected.version));
#endif
  }
  CHECK(read_bytes() == source_bytes);
}

TEST_CASE("PsiFormer CPU pathwise gradient differentiates the complete reweighted finite objective",
          "[wavefunction][psiformer][complex_wf][complex_training]")
{
  using namespace wftrain;
  auto lattice = cell();
  lattice.LR_dim_cutoff = 15;
  lattice.reset();
  const SimulationCell simulation_cell(lattice);
  auto p = electrons(simulation_cell), clone_p = electrons(simulation_cell);
  for (auto* target : {&p, &clone_p})
  {
    const int mass = target->getSpeciesSet().findAttribute("mass");
    target->getSpeciesSet()(mass, 0) = .7;
    target->getSpeciesSet()(mass, 1) = 1.9;
    const int charge = target->getSpeciesSet().addAttribute("charge");
    target->getSpeciesSet()(charge, 0) = target->getSpeciesSet()(charge, 1) = -1;
    target->resetGroups();
    target->createSK();
  }
  clone_p.R[1][2] += .073;
  clone_p.update();
  PsiFormerWaveFunctionBuilder::PSetMap pool;
  pool.emplace("ion0", std::make_unique<ParticleSet>(ions(simulation_cell)));
  auto& source = *pool.at("ion0");
  source.createSK();
  RuntimeOptions options;
  TrialWaveFunction trial(options);
  auto owner = std::make_unique<PsiFormerComplexWF>("pathwise", parameters(), p, source, clone_cap,
                                                  std::array<double, 3>{.17, -.11, .07});
  auto& component = *owner;
  trial.addComponent(std::move(owner));
  auto clone_trial = trial.makeClone(clone_p);
  auto& clone = dynamic_cast<PsiFormerComplexWF&>(*clone_trial->getOrbitals()[0]);
  std::array<TrialWaveFunction*, 2> wavefunctions{&trial, clone_trial.get()};
  std::array<ParticleSet*, 2> particles{&p, &clone_p};
  std::array<std::unique_ptr<QMCHamiltonian>, 2> hamiltonians;
  for (std::size_t sample = 0; sample < 2; ++sample)
  {
    Libxml2Document input;
    REQUIRE(input.parseFromString("<hamiltonian>" + coulombTerms + "</hamiltonian>"));
    HamiltonianFactory factory("pathwise_coulomb", *particles[sample], pool,
                              OptionalRef<TrialWaveFunction>(*wavefunctions[sample]), OHMMS::Controller);
    REQUIRE(factory.put(input.getRoot()));
    hamiltonians[sample] = factory.releaseHamiltonian();
  }
  source.update();
  // Refresh the actual accepted TWF/H pair before borrowing its complete complex energy.
  const auto observe = [&](std::size_t sample) {
    auto& target = *particles[sample];
    target.update();
    wavefunctions[sample]->evaluateLog(target);
    const double real_energy = hamiltonians[sample]->evaluate(*wavefunctions[sample], target);
    auto& actual = sample == 0 ? component : clone;
    const auto complete = actual.observeCoulombEnergy(target, hamiltonians[sample]->getLocalPotential());
    CHECK(complete.local_energy.real() == Approx(real_energy).epsilon(2e-11));
    return std::make_pair(C{wavefunctions[sample]->getLogPsi(), wavefunctions[sample]->getPhase()},
                          complete.local_energy);
  };
  const auto original = component.snapshotParameters();
  const auto identity = component.trainingConfigurationFingerprint();
  CHECK(clone.trainingConfigurationFingerprint() == identity);
  std::array<C, 2> log_reference;
  PreparedEnergyGradientBatch batch;
  batch.weights = {.4, 1.3};
  batch.local_energy_term_mask = localEnergyTermBit(LocalEnergyTerm::KINETIC);
  for (std::size_t sample = 0; sample < 2; ++sample)
  {
    const auto result = observe(sample);
    log_reference[sample] = result.first;
    batch.local_energies.push_back(result.second);
  }
  CHECK(std::abs(batch.local_energies[0].imag()) > 1e-8);
  RefVectorWithLeader<WaveFunctionComponent> components(component);
  components.push_back(component);
  components.push_back(clone);
  RefVectorWithLeader<ParticleSet> particle_refs(p);
  particle_refs.push_back(p);
  particle_refs.push_back(clone_p);
  batch.derivative_operator = component.makeBudgetedStreamingDerivativeOperator(components, particle_refs,
      0, 0, 5, component.streamingDerivativeRequiredBytes(2));
  std::vector<PreparedEnergyGradientBatch> batches;
  batches.push_back(std::move(batch));
  PreparedEnergyGradientProducer producer(std::move(batches));
  EnergyGradientAccumulator accumulator(component.parameterSchema(), original.version,
                                        EnergyGradientEstimator::PATHWISE_LOCAL_ENERGY);
  producer.accumulate(original, accumulator);
  DistributedParameterReduction reduction(*OHMMS::Controller);
  reduction.reduce(accumulator);
  const auto gradient = accumulator.finalize();
  std::vector<double> direction(original.values.size());
  double analytic = 0;
  std::size_t index = 0;
  for (const auto& block : component.parameterSchema().blocks())
  {
    direction[block.offset] = (++index % 2) ? .08 : -.05;
    analytic += direction[block.offset] * gradient.gradient[block.offset];
  }
  // Moving the complete normalized finite objective differentiates BOTH weights and local energies.
  const auto objective = [&](double displacement) {
    auto candidate = component.snapshotParameters();
    candidate.values = original.values;
    for (std::size_t parameter = 0; parameter < direction.size(); ++parameter)
      candidate.values[parameter] += displacement * direction[parameter];
    component.publishParameters(candidate, candidate.version);
    double numerator = 0, denominator = 0;
    for (std::size_t sample = 0; sample < 2; ++sample)
    {
      const auto result = observe(sample);
      const double weight = (sample == 0 ? .4 : 1.3) *
          std::exp(2 * (result.first.real() - log_reference[sample].real()));
      numerator += weight * result.second.real();
      denominator += weight;
    }
    CHECK(component.trainingConfigurationFingerprint() == identity);
    return numerator / denominator;
  };
  const double epsilon = 2e-5;
  const double plus = objective(epsilon), minus = objective(-epsilon);
  CHECK(analytic == Approx((plus - minus) / (2 * epsilon)).margin(3e-5));
  auto restored = component.snapshotParameters();
  restored.values = original.values;
  component.publishParameters(restored, restored.version);
  CHECK(component.trainingConfigurationFingerprint() == identity);
  auto other_p = electrons(simulation_cell);
  PsiFormerComplexWF wrong_twist("pathwise", parameters(), other_p, source, clone_cap, {.18, -.11, .07});
  CHECK(wrong_twist.trainingConfigurationFingerprint() != identity);
}
#ifdef ENABLE_CUDA
TEST_CASE("PsiFormer complex C2 score lanes preserve weighted partition sums",
          "[wavefunction][psiformer][complex_gpu_score_crowds]")
{
  using namespace wftrain;
  namespace dev = ps::device;
  int devices = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices)
  { WARN("NOT_RUN: C2 scores require a HIP device"); return; }
  REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
  { WARN("NOT_RUN: C2 scores require a CUDA device"); return; }
  REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
  if (!dev::configuredDeviceMemoryAdmission().limit_bytes)
  { WARN("NOT_RUN: C2 scores require PSIFORMER_GPU_AGGREGATE_DEVICE_BYTES"); return; }
  ScopedComplexGpuControls controls;
  controls.set("PSIFORMER_GPU_ACTIVE_INFERENCE_CROWDS", "2");
  const SimulationCell simulation_cell(cell());
  auto p0 = electrons(simulation_cell), p1 = electrons(simulation_cell), p2 = electrons(simulation_cell);
  p1.R[1][0] += .017; p1.update(); p2.R[2][1] -= .031; p2.update();
  auto source = ions(simulation_cell);
  const auto initialized = parameters();
  const std::array<double, 3> twist{.17, -.11, .07};
  RuntimeOptions options;
  TrialWaveFunction trial(options);
  auto owner = std::make_unique<PsiFormerComplexWF>("score_crowds", initialized, p0, source, clone_cap, twist,
                                                  64 * 1024 * 1024);
  auto& gpu = *owner;
  trial.addComponent(std::move(owner));
  auto trial1 = trial.makeClone(p1), trial2 = trial.makeClone(p2);
  auto& gpu1 = dynamic_cast<PsiFormerComplexWF&>(*trial1->getOrbitals()[0]);
  RefVectorWithLeader<TrialWaveFunction> trials0(trial, {trial}), trials1(*trial1, {*trial1, *trial2});
  RefVectorWithLeader<ParticleSet> particles0(p0, {p0}), particles1(p1, {p1, p2});
  constexpr auto scope = PsiFormerComplexWF::GpuDerivativeScope::SCORE_ECP_ONLY;
  const auto r0 = gpu.gpuDerivativeRequirements(1, 1, 1, 7, scope);
  const auto r1 = gpu1.gpuDerivativeRequirements(2, 2, 1, 7, scope);
  auto first = gpu.makeBudgetedGpuDerivativeOperator(trials0, particles0, 0, 0, 1, 1, 7,
      r0.device_bytes, r0.host_bytes, r0.matrix_work, scope);
  auto second = gpu1.makeBudgetedGpuDerivativeOperator(trials1, particles1, 1, 1, 2, 1, 7,
      r1.device_bytes, r1.host_bytes, r1.matrix_work, scope);
  CHECK(first->storageDiagnostics().shared_workspace_identity != second->storageDiagnostics().shared_workspace_identity);
  CHECK_THROWS(gpu.gpuDerivativeRequirements(1, 1, 1, 7));
  CHECK_THROWS(gpu.makeBudgetedGpuDerivativeOperator(trials0, particles0, 2, 2, 1, 1, 7,
      r0.device_bytes, r0.host_bytes, r0.matrix_work, scope));
  PsiFormerComplexWF cpu("score_crowds", initialized, p0, source, clone_cap, twist);
  auto cpu1 = cpu.makeClone(p1), cpu2 = cpu.makeClone(p2);
  RefVectorWithLeader<WaveFunctionComponent> components(cpu, {cpu, *cpu1, *cpu2});
  RefVectorWithLeader<ParticleSet> all(p0, {p0, p1, p2});
  auto oracle = cpu.makeBudgetedStreamingDerivativeOperator(components, all, 0, 0, 7,
      cpu.streamingDerivativeRequiredBytes(3));
  const std::vector<C> coefficients{{.7, -.2}, {-.3, .1}, {.4, .05}};
  const auto contract = [&](const StreamingDerivativeOperator& op, std::size_t batch, std::size_t offset) {
#ifdef QMC_CUDA2HIP
    dev::checkRuntime(hipSetDevice(ordinal), "bind complex score worker");
#else
    dev::checkRuntime(cudaSetDevice(ordinal), "bind complex score worker");
#endif
    const CoefficientView weights{op.parameterSchema().providerId(), op.parameterSchema().fingerprint(),
        op.parameterVersion(), batch, offset, {coefficients.data() + offset, op.sampleCount()}};
    const VJPCoefficientChannel channel{"score", DerivativeProduct::SCORE_VJP, weights, 0};
    ComplexStreamingSink sink;
    op.applyVJPs({&channel, 1}, DerivativeAdjoint::HERMITIAN, sink);
    return sink.values[0];
  };
  // Uneven 1+2 crowd tails use separate queues; the complete weighted population
  // has the same contraction as the independent CPU three-sample oracle.
  auto left = std::async(std::launch::async, [&] { return contract(*first, 0, 0); });
  auto right = std::async(std::launch::async, [&] { return contract(*second, 1, 1); });
  const auto expected = contract(*oracle, 0, 0);
  const auto a = left.get(), b = right.get();
  REQUIRE(a.size() == expected.size()); REQUIRE(b.size() == expected.size());
  for (std::size_t i = 0; i < a.size(); ++i) close(a[i] + b[i], expected[i], 3e-9);
  auto candidate = gpu.snapshotParameters(); candidate.values[0] += .001;
  const auto q = gpu.gpuParameterPublicationRequirements();
  gpu.publishBudgetedGpuParameters(candidate, candidate.version, q.device_bytes, q.host_bytes);
  CHECK_THROWS(contract(*first, 0, 0)); CHECK_THROWS(contract(*second, 1, 1));
}

TEST_CASE("PsiFormer immutable GPU analysis streams preserve complete physical products",
          "[wavefunction][psiformer][complex_gpu_derivatives]")
{
  using namespace wftrain;
  namespace dev = ps::device;
  int devices = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices)
  {
    WARN("NOT_RUN: GPU derivative binding requires an available HIP device");
    return;
  }
  REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
  {
    WARN("NOT_RUN: GPU derivative binding requires an available CUDA device");
    return;
  }
  REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
  ScopedComplexGpuControls controls;
  std::size_t samples = 2;
  SECTION("default T1 inference and B1 products") {}
  SECTION("dense T2 inference W3 tail with serialized B1 products")
  {
    samples = 3;
    controls.set("PSIFORMER_GPU_COMPLEX_TILE_CAPACITY", "2");
  }
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell, 3, 1), clone_p = electrons(simulation_cell, 3, 1),
       tail_p = electrons(simulation_cell, 3, 1);
  auto source = ions(simulation_cell);
  for (auto* target : {&p, &clone_p, &tail_p})
  {
    const int mass = target->getSpeciesSet().findAttribute("mass");
    target->getSpeciesSet()(mass, 0) = .7;
    target->getSpeciesSet()(mass, 1) = 1.9;
    target->resetGroups();
  }
  clone_p.R[1][2] += .037;
  clone_p.update();
  tail_p.R[2][0] -= .043;
  tail_p.update();
  RuntimeOptions options;
  TrialWaveFunction trial(options);
  const std::array<double, 3> twist{.17, -.11, .07};
  auto initialized = parameters();
  initialized.model_shape.spin_up_electrons = 3;
  initialized.model_shape.spin_down_electrons = 1;
  auto owner = std::make_unique<PsiFormerComplexWF>("analysis", initialized, p, source, clone_cap, twist,
                                                  64 * 1024 * 1024);
  auto& component = *owner;
  trial.addComponent(std::move(owner));
  auto clone_trial = trial.makeClone(clone_p);
  auto tail_trial = trial.makeClone(tail_p);
  RefVectorWithLeader<TrialWaveFunction> trials(trial);
  trials.push_back(trial);
  trials.push_back(*clone_trial);
  if (samples == 3) trials.push_back(*tail_trial);
  RefVectorWithLeader<ParticleSet> particles(p);
  particles.push_back(p);
  particles.push_back(clone_p);
  if (samples == 3) particles.push_back(tail_p);
  const auto r0 = component.gpuDerivativeRequirements(0, samples, 3, 7);
  const auto r2 = component.gpuDerivativeRequirements(samples, samples, 3, 7);
  std::printf("GPU derivative admission R0_device=%zu R0_host=%zu R2_device=%zu R2_host=%zu matrix_work=%zu\n",
      r0.device_bytes, r0.host_bytes, r2.device_bytes, r2.host_bytes, r2.matrix_work);
  CHECK(r2.device_bytes == r0.device_bytes);
  CHECK(r2.host_bytes == r0.host_bytes + 32 * samples * 4);
  CHECK(component.gpuDerivativeRequirements(4, samples, 3, 7).host_bytes == r0.host_bytes + 32 * 4 * 4);
  auto& metrics = dev::deviceMemoryMetrics(ordinal);
  const auto before = metrics.snapshot();
  CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, samples, 3, 7,
      r2.device_bytes - 1, r2.host_bytes, r2.matrix_work));
  CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, samples, 3, 7,
      r2.device_bytes, r2.host_bytes - 1, r2.matrix_work));
  CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, samples, 3, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work - 1));
  const auto mass = p.getSpeciesSet().findAttribute("mass");
  p.getSpeciesSet()(mass, 0) = 0;
  p.resetGroups();
  CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, samples, 3, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work));
  p.getSpeciesSet()(mass, 0) = .7;
  p.resetGroups();
  p.setSpinor(true);
  CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, samples, 3, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work));
  p.setSpinor(false);
  const auto last_spin = p.spins[3];
  p.spins.resize(3);
  CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, samples, 3, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work));
  p.spins.resize(4);
  p.spins[3] = last_spin;
  const auto spin = p.spins[0];
  p.spins[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, samples, 3, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work));
  p.spins[0] = spin;
  const int charge = source.getSpeciesSet().findAttribute("charge");
  const auto source_charge = source.getSpeciesSet()(charge, 0);
  const auto source_coordinate = source.R[0][0];
  for (int field = 0; field < 2; ++field)
  {
    auto& scalar = field ? source.R[0][0] : source.getSpeciesSet()(charge, 0);
    scalar = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, samples, 3, 7,
        r2.device_bytes, r2.host_bytes, r2.matrix_work));
    scalar = field ? source_coordinate : source_charge;
  }
  CHECK(metrics.snapshot().allocation_calls == before.allocation_calls);
  auto stream = component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, samples, 3, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work);
  const auto diagnostic = stream->storageDiagnostics();
  CHECK(diagnostic.retained_numeric_bytes == r2.device_bytes + r2.host_bytes);
  CHECK(diagnostic.parameter_scratch_bytes == 112 * component.parameterSchema().parameterCount() + 32 * 7);
  CHECK(diagnostic.complex_parameter_vectors == 5);
  const auto* admitted_image = static_cast<const dev::PreparedDeviceModel*>(component.numericStorage().shared_image_identity);
  const auto retained_cost = dev::estimateComplexProductsCost(admitted_image->plan(), 3, samples, 3, 7);
  const auto chunked_cost = dev::estimateComplexProductsCost(admitted_image->plan(), 3, samples, 3, 7,
      dev::ComplexProductsHostOutput::CHUNKED);
  CHECK(chunked_cost.device_bytes == retained_cost.device_bytes);
  CHECK(retained_cost.host_bytes - chunked_cost.host_bytes == 48 * component.parameterSchema().parameterCount() - 16 * 7);
  CHECK(diagnostic.evaluator_workspace_bytes + diagnostic.parameter_scratch_bytes == diagnostic.shared_workspace_bytes);
  CHECK(diagnostic.shared_workspace_bytes == r0.device_bytes + r0.host_bytes);
  CHECK(stream->capabilities().supports(DerivativeProduct::SCORE_JVP));
  CHECK(stream->capabilities().supports(DerivativeAdjoint::HERMITIAN));
  CHECK(component.structuredParameterProvider() == nullptr);
  const auto snapshot = component.snapshotParameters();
  CHECK_THROWS(component.publishParameters(snapshot, snapshot.version));

  // The independently qualified CPU two-tape owner supplies complete complex rows, not Re-only forces.
  PsiFormerComplexWF cpu("analysis", initialized, p, source, clone_cap, twist);
  CHECK(component.neuralWeightBlocks() == cpu.neuralWeightBlocks());
  auto cpu_clone = cpu.makeClone(clone_p);
  auto cpu_tail = cpu.makeClone(tail_p);
  RefVectorWithLeader<WaveFunctionComponent> components(cpu);
  components.push_back(cpu);
  components.push_back(*cpu_clone);
  if (samples == 3) components.push_back(*cpu_tail);
  // The sampler's numerical graph really runs T2 plus tail1 before and after
  // publication; Products below independently keeps its existing B1 owners.
  const auto check_dense_full = [&] {
    if (samples != 3) return;
    RefVectorWithLeader<WaveFunctionComponent> gpu_components(component);
    RefVector<ParticleSet::ParticleGradient> gradients;
    RefVector<ParticleSet::ParticleLaplacian> laplacians;
    for (std::size_t i = 0; i < samples; ++i)
    {
      gpu_components.push_back(*trials[i].getOrbitals()[0]);
      particles[i].G = 0; particles[i].L = 0;
      gradients.push_back(particles[i].G); laplacians.push_back(particles[i].L);
    }
    ResourceCollection resources("dense inference in serialized training");
    component.createResource(resources);
    ResourceCollectionTeamLock<WaveFunctionComponent> loan(resources, gpu_components);
    component.mw_evaluateLog(gpu_components, particles, gradients, laplacians);
    for (std::size_t i = 0; i < samples; ++i)
    {
      const auto gradient = particles[i].G;
      const auto laplacian = particles[i].L;
      close(gpu_components[i].get_log_value(), evaluate(components[i], particles[i]));
      for (std::size_t e = 0; e < 4; ++e)
      {
        for (int d = 0; d < 3; ++d) close(gradient[e][d], particles[i].G[e][d]);
        close(laplacian[e], particles[i].L[e]);
      }
    }
  };
  check_dense_full();
  auto oracle = cpu.makeBudgetedStreamingDerivativeOperator(components, particles, 7, 11, 7,
      cpu.streamingDerivativeRequiredBytes(samples));
  CHECK_THROWS(cpu.gpuDerivativeRequirements(2, 2, 3, 7));
  const auto& schema = stream->parameterSchema();
  std::vector<double> weights{.4, 1.3};
  if (samples == 3) weights.push_back(.7);
  std::vector<C> energies(samples);
  evaluate(cpu, p);
  evaluate(*cpu_clone, clone_p);
  energies[0] = cpu.observeCoulombEnergy(p, -1.3).local_energy;
  energies[1] = static_cast<PsiFormerComplexWF&>(*cpu_clone).observeCoulombEnergy(clone_p, .7).local_energy;
  if (samples == 3)
  {
    evaluate(*cpu_tail, tail_p);
    energies[2] = static_cast<PsiFormerComplexWF&>(*cpu_tail).observeCoulombEnergy(tail_p, -.4).local_energy;
  }
  CHECK(std::abs(energies[0].imag()) + std::abs(energies[1].imag()) > 1e-8);
  // Compare the actual complete PATHWISE producer, including dEL=dK, not a score-only shortcut.
  const auto complete_force = [&](std::unique_ptr<StreamingDerivativeOperator> derivative) {
    PreparedEnergyGradientBatch batch;
    batch.derivative_operator = std::move(derivative);
    batch.weights.assign(weights.begin(), weights.end());
    batch.local_energies.assign(energies.begin(), energies.end());
    batch.local_energy_term_mask = localEnergyTermBit(LocalEnergyTerm::KINETIC);
    std::vector<PreparedEnergyGradientBatch> batches;
    batches.push_back(std::move(batch));
    PreparedEnergyGradientProducer producer(std::move(batches));
    EnergyGradientAccumulator accumulator(schema, snapshot.version, EnergyGradientEstimator::PATHWISE_LOCAL_ENERGY);
    producer.accumulate(snapshot, accumulator);
    DistributedParameterReduction reduction(*OHMMS::Controller);
    reduction.reduce(accumulator);
    return accumulator.finalize().gradient;
  };
  const auto cpu_force = complete_force(cpu.makeBudgetedStreamingDerivativeOperator(components, particles, 7, 11, 7,
      cpu.streamingDerivativeRequiredBytes(samples)));
  const auto gpu_force = complete_force(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, samples, 3, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work));
  for (std::size_t i = 0; i < schema.parameterCount(); ++i)
    CHECK(gpu_force[i] == Catch::Approx(cpu_force[i]).margin(3e-9));
  std::vector<C> direction(schema.parameterCount()), phase_direction(schema.parameterCount());
  std::size_t direction_block = 0;
  for (const auto& block : schema.blocks())
  {
    direction[block.offset] = (++direction_block % 2) ? .08 : -.05;
    if (block.id.find("/w_imag") != std::string::npos)
      phase_direction[block.offset] = .6;
  }
  ComplexSampleSink cpu_jvp, gpu_jvp;
  bool jvp_callback_ran = false;
  gpu_jvp.callback = [&] {
    if (!jvp_callback_ran) { jvp_callback_ran = true; evaluate(component, p); }
  };
  const auto check_jvp = [&](const std::vector<C>& values) {
    cpu_jvp.reset(); gpu_jvp.reset();
    const StructuredParameterVectorConstView tangent(schema, snapshot.version, {values.data(), values.size()});
    oracle->applyScoreJVP(tangent, cpu_jvp);
    stream->applyScoreJVP(tangent, gpu_jvp);
    for (std::size_t i = 0; i < samples; ++i) close(gpu_jvp.values[i], cpu_jvp.values[i], 3e-9);
  };
  check_jvp(std::vector<C>(direction.size()));
  for (const auto value : gpu_jvp.values) CHECK(value == C{});
  check_jvp(direction);
  check_jvp(phase_direction);
  CHECK(jvp_callback_ran);
  CHECK(std::abs(gpu_jvp.values[0].imag()) + std::abs(gpu_jvp.values[1].imag()) > 1e-8);
  // The unchanged host SR operator consumes these REAL-tangent GPU products.
  const StochasticReconfigurationBatch gpu_batch{stream.get(), {weights.data(), weights.size()}};
  const StochasticReconfigurationBatch cpu_batch{oracle.get(), {weights.data(), weights.size()}};
  StochasticReconfigurationOperator gpu_metric(schema, snapshot.version, {&gpu_batch, 1});
  StochasticReconfigurationOperator cpu_metric(schema, snapshot.version, {&cpu_batch, 1});
  const std::array<const std::vector<C>*, 2> directions{&direction, &phase_direction};
  std::array<std::vector<C>, 2> actions{std::vector<C>(direction.size()), std::vector<C>(direction.size())};
  for (std::size_t column = 0; column < 2; ++column)
  {
    std::vector<C> expected_action(direction.size());
    const StructuredParameterVectorConstView tangent(schema, snapshot.version, {directions[column]->data(), direction.size()});
    gpu_metric.apply(tangent, {actions[column].data(), direction.size()});
    cpu_metric.apply(tangent, {expected_action.data(), direction.size()});
    double damped_quadratic = 0;
    for (std::size_t i = 0; i < direction.size(); ++i)
    {
      close(actions[column][i], expected_action[i], 3e-9);
      damped_quadratic += (*directions[column])[i].real() * actions[column][i].real() +
          .1 * std::norm((*directions[column])[i]);
    }
    CHECK(damped_quadratic > 0);
  }
  double left = 0, right = 0;
  for (std::size_t i = 0; i < direction.size(); ++i)
  {
    left += direction[i].real() * actions[1][i].real();
    right += phase_direction[i].real() * actions[0][i].real();
  }
  CHECK(left == Catch::Approx(right).margin(3e-9));
  IdentityPreconditioner preconditioner(schema, snapshot.version);
  StochasticReconfigurationUpdateControl sr_control;
  sr_control.learning_rate = 1e-4;
  sr_control.initial_damping = .1;
  sr_control.maximum_damping_attempts = 1;
  sr_control.krylov.relative_tolerance = 1e-7;
  sr_control.krylov.maximum_iterations = 6;
  sr_control.krylov.maximum_operator_applications = 10;
  StochasticReconfigurationUpdateRule gpu_update(gpu_metric, preconditioner, sr_control);
  StochasticReconfigurationUpdateRule cpu_update(cpu_metric, preconditioner, sr_control);
  const ParameterGradientView gpu_objective{schema.fingerprint(), snapshot.version, ReductionDomain::GLOBAL,
      {gpu_force.data(), gpu_force.size()}};
  const ParameterGradientView cpu_objective{schema.fingerprint(), snapshot.version, ReductionDomain::GLOBAL,
      {cpu_force.data(), cpu_force.size()}};
  const auto gpu_sr = gpu_update.propose(schema, snapshot, gpu_objective);
  const auto cpu_sr = cpu_update.propose(schema, snapshot, cpu_objective);
  for (std::size_t i = 0; i < direction.size(); ++i)
    CHECK(gpu_sr.values[i] == Catch::Approx(cpu_sr.values[i]).margin(3e-9));
  std::array<std::vector<C>, 3> coefficients{{{{.3, -.2}, {-.4, .1}},
      {{.7, .15}, {.2, -.35}}, {{-.6, .4}, {.8, -.1}}}};
  if (samples == 3)
    for (auto& channel : coefficients) channel.push_back({.11, -.23});
  std::array<VJPCoefficientChannel, 3> channels;
  for (std::size_t c = 0; c < channels.size(); ++c)
    channels[c] = {c == 0 ? "score0" : c == 1 ? "kinetic" : "score2",
        c == 1 ? DerivativeProduct::LOCAL_ENERGY_VJP : DerivativeProduct::SCORE_VJP,
        {schema.providerId(), schema.fingerprint(), stream->parameterVersion(), 7, 11,
          {coefficients[c].data(), samples}}, c == 1 ? localEnergyTermBit(LocalEnergyTerm::KINETIC) : 0};
  ComplexStreamingSink expected, actual;
  oracle->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, expected);
  // Live coordinates may change after capture; callbacks may safely re-enter ordinary GPU inference.
  p.R[0][0] += .019;
  p.update();
  bool callback_ran = false;
  actual.callback = [&] {
    if (!callback_ran)
    {
      callback_ran = true;
      evaluate(component, p);
    }
  };
  const auto warm = metrics.snapshot();
  stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual);
  CHECK(callback_ran);
  for (std::size_t c = 0; c < 3; ++c)
    for (std::size_t parameter = 0; parameter < schema.parameterCount(); ++parameter)
      close(actual.values[c][parameter], expected.values[c][parameter], 3e-9);
  actual.reset();
  stream->applyVJPs({channels.data(), 1}, DerivativeAdjoint::TRANSPOSE, actual);
  for (std::size_t parameter = 0; parameter < schema.parameterCount(); ++parameter)
    close(actual.values[0][parameter], expected.values[0][parameter], 3e-9);
  CHECK(metrics.snapshot().allocation_calls == warm.allocation_calls);
  CHECK(metrics.snapshot().free_calls == warm.free_calls);
  CHECK(stream->storageDiagnostics().storage_fingerprint == diagnostic.storage_fingerprint);
  gpu_jvp.reset();
  gpu_jvp.fail_next = true;
  CHECK_THROWS(stream->applyScoreJVP({schema, snapshot.version, {direction.data(), direction.size()}}, gpu_jvp));
  CHECK(gpu_jvp.state() == DerivativeSinkState::POISONED);
  check_jvp(direction);
  auto second = component.makeBudgetedGpuDerivativeOperator(trials, particles, 8, 13, samples, 3, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work);
  CHECK(second->storageDiagnostics().shared_workspace_identity == diagnostic.shared_workspace_identity);
  CHECK(second->storageDiagnostics().retained_numeric_bytes + diagnostic.retained_numeric_bytes -
      diagnostic.shared_workspace_bytes == r0.device_bytes + r0.host_bytes + 32 * 2 * samples * 4);
  CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, samples, 2, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work));
  actual.reset();
  actual.fail_next = true;
  CHECK_THROWS(stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual));
  CHECK(actual.state() == DerivativeSinkState::POISONED);
  actual.reset();
  CHECK_NOTHROW(stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual));
  actual.reset();
  ComplexStreamingSink expected_hermitian;
  oracle->applyVJPs({channels.data(), 3}, DerivativeAdjoint::HERMITIAN, expected_hermitian);
  stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::HERMITIAN, actual);
  for (std::size_t c = 0; c < 3; ++c)
    for (std::size_t i = 0; i < schema.parameterCount(); ++i)
      close(actual.values[c][i], expected_hermitian.values[c][i], 3e-9);
  C sample_dot{}, parameter_dot{};
  for (std::size_t sample = 0; sample < samples; ++sample)
    sample_dot += std::conj(coefficients[0][sample]) * gpu_jvp.values[sample];
  for (std::size_t i = 0; i < direction.size(); ++i)
    parameter_dot += std::conj(actual.values[0][i]) * direction[i].real();
  close(sample_dot, parameter_dot, 3e-9);
  actual.reset();
  channels[1].local_energy_term_mask |= localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);
  CHECK_THROWS(stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual));
  channels[1].local_energy_term_mask = localEnergyTermBit(LocalEnergyTerm::KINETIC);
  ++channels[0].coefficients.parameter_version;
  CHECK_THROWS(stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual));
  --channels[0].coefficients.parameter_version;
  const auto finite_coefficient = coefficients[0][1];
  coefficients[0][1] = {std::numeric_limits<double>::infinity(), 0};
  CHECK_THROWS(stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual));
  coefficients[0][1] = finite_coefficient;
  const auto source_position = source.R[0];
  source.R[0][0] += .01;
  CHECK_THROWS(stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual));
  source.R[0] = source_position;
  for (int field = 0; field < 2; ++field)
  {
    auto& scalar = field ? source.R[0][0] : source.getSpeciesSet()(charge, 0);
    scalar = std::numeric_limits<double>::quiet_NaN();
    actual.reset();
    CHECK_THROWS(stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual));
    scalar = field ? source_coordinate : source_charge;
  }
  actual.reset();
  p.makeMove(0, ParticleSet::PosType{.01, 0, 0});
  component.ratio(p, 0);
  CHECK_THROWS(stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual));
  gpu_jvp.reset();
  CHECK_THROWS(stream->applyScoreJVP({schema, snapshot.version, {direction.data(), direction.size()}}, gpu_jvp));
  component.restore(0);
  p.rejectMove(0);
  actual.reset();
  CHECK_NOTHROW(stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual));
  const auto publication = component.gpuParameterPublicationRequirements();
  // Publication on the FIRST bounded output chunk must leave all remaining
  // old-image chunks valid, with no model/lane lock held through this callback.
  bool callback_published = false;
  actual.reset();
  actual.callback = [&] {
    if (callback_published) return;
    callback_published = true;
    CHECK(component.publishBudgetedGpuParameters(gpu_sr, snapshot.version, publication.device_bytes, publication.host_bytes)
        == snapshot.version + 1);
    CHECK(cpu.publishParameters(cpu_sr, snapshot.version) == snapshot.version + 1);
  };
  stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::HERMITIAN, actual);
  CHECK(callback_published);
  for (std::size_t c = 0; c < 3; ++c)
    for (std::size_t i = 0; i < schema.parameterCount(); ++i)
      close(actual.values[c][i], expected_hermitian.values[c][i], 3e-9);
  check_dense_full();
  auto current = component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, samples, 3, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work);
  auto current_cpu = cpu.makeBudgetedStreamingDerivativeOperator(components, particles, 7, 11, 7,
      cpu.streamingDerivativeRequiredBytes(samples));
  for (auto& channel : channels) channel.coefficients.parameter_version = snapshot.version + 1;
  actual.reset(); expected.reset();
  actual.callback = {};
  current->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual);
  current_cpu->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, expected);
  for (std::size_t c = 0; c < 3; ++c)
    for (std::size_t i = 0; i < schema.parameterCount(); ++i)
      close(actual.values[c][i], expected.values[c][i], 3e-9);
  for (auto& channel : channels) channel.coefficients.parameter_version = snapshot.version;
  trial.addComponent(component.makeClone(p));
  CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, samples, 3, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work));
  gpu_update.proposalAccepted(schema, snapshot, gpu_objective);
  cpu_update.proposalAccepted(schema, snapshot, cpu_objective);
  CHECK_FALSE(gpu_update.hasLiveProposal());
  gpu_jvp.reset();
  CHECK_THROWS(stream->applyScoreJVP({schema, snapshot.version, {direction.data(), direction.size()}}, gpu_jvp));
  actual.reset();
  CHECK_THROWS(stream->applyVJPs({channels.data(), 1}, DerivativeAdjoint::HERMITIAN, actual));
}

TEST_CASE("PsiFormer private GPU candidates isolate mutable lanes and preserve source products",
          "[wavefunction][psiformer][complex_gpu_candidate_isolation]")
{
  using namespace wftrain;
  namespace dev = ps::device;
  int devices = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices) { WARN("NOT_RUN: GPU isolation needs HIP"); return; }
  REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) { WARN("NOT_RUN: GPU isolation needs CUDA"); return; }
  REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
  ScopedComplexGpuControls controls;
  controls.target(3, 64 * 1024 * 1024, 1000000000);
  // Target-K uses the existing B1 lane, not the separately admitted dense profile.
  controls.set("PSIFORMER_GPU_COMPLEX_TILE_CAPACITY", "1");
  auto& metrics = dev::deviceMemoryMetrics(ordinal);
  const auto outside = metrics.snapshot();
  {
    const SimulationCell simulation_cell(cell());
    auto p = electrons(simulation_cell), sibling_p = electrons(simulation_cell);
    auto private_p = electrons(simulation_cell), cpu_p = electrons(simulation_cell), source = ions(simulation_cell);
    const auto initialized = parameters();
    const std::array<double, 3> twist{.17, -.11, .07};
    RuntimeOptions options;
    TrialWaveFunction wavefunction(options);
    auto owner = std::make_unique<PsiFormerComplexWF>("private_gpu", initialized, p, source, clone_cap,
                                                     twist, 64 * 1024 * 1024);
    auto& gpu = *owner;
    wavefunction.addComponent(std::move(owner));
    auto sibling_owner = gpu.makeClone(sibling_p), private_owner = gpu.makeClone(private_p);
    auto& sibling = dynamic_cast<PsiFormerComplexWF&>(*sibling_owner);
    auto& candidate_wf = dynamic_cast<PsiFormerComplexWF&>(*private_owner);
    PsiFormerComplexWF cpu("private_gpu", initialized, cpu_p, source, clone_cap, twist);
    const auto original = gpu.snapshotParameters();
    const auto original_log = evaluate(gpu, p);
    const auto original_g = p.G;
    const auto original_l = p.L;
    evaluate(sibling, sibling_p);
    evaluate(candidate_wf, private_p);
    RefVectorWithLeader<TrialWaveFunction> wavefunctions(wavefunction, {wavefunction});
    RefVectorWithLeader<ParticleSet> particles(p, {p});
    const auto derivative = gpu.gpuDerivativeRequirements(1, 1, 3, 7);
    auto retained = gpu.makeBudgetedGpuDerivativeOperator(wavefunctions, particles, 0, 0, 1, 3, 7,
        derivative.device_bytes, derivative.host_bytes, derivative.matrix_work);
    const auto score = [&](StreamingDerivativeOperator& stream) {
      const C one{1};
      const auto& schema = stream.parameterSchema();
      const VJPCoefficientChannel channel{"score", DerivativeProduct::SCORE_VJP,
          {schema.providerId(), schema.fingerprint(), stream.parameterVersion(), 0, 0, {&one, 1}}, 0};
      ComplexStreamingSink sink;
      stream.applyVJPs({&channel, 1}, DerivativeAdjoint::TRANSPOSE, sink);
      return sink.values[0];
    };
    const auto original_score = score(*retained);
    const auto storage = gpu.numericStorage();
    const auto isolation = candidate_wf.gpuCandidateIsolationRequirements();
    REQUIRE(isolation.device_bytes > 0);
    REQUIRE(isolation.host_bytes > 0);
    const auto before = metrics.snapshot();
    CHECK_THROWS(candidate_wf.isolateBudgetedGpuParameterState(isolation.device_bytes - 1, isolation.host_bytes));
    CHECK_THROWS(candidate_wf.isolateBudgetedGpuParameterState(isolation.device_bytes, isolation.host_bytes - 1));
    CHECK(metrics.snapshot().allocation_calls == before.allocation_calls);
    CHECK(metrics.snapshot().uploaded_bytes == before.uploaded_bytes);
    CHECK(metrics.snapshot().device_copied_bytes == before.device_copied_bytes);
    CHECK(candidate_wf.numericStorage().shared_image_identity == storage.shared_image_identity);
    CHECK(candidate_wf.snapshotParameters().values == original.values);
    CHECK(candidate_wf.snapshotParameters().version == original.version);
    CHECK(gpu.snapshotParameters().values == original.values);
    CHECK(sibling.snapshotParameters().values == original.values);
    // Frozen preparation must survive later invalid process controls; no initial P image is copied.
    controls.set("PSIFORMER_GPU_VGL_TARGETS", "not-a-number");
    controls.set("PSIFORMER_GPU_COMPLEX_TILE_CAPACITY", "not-a-number");
    candidate_wf.isolateBudgetedGpuParameterState(isolation.device_bytes, isolation.host_bytes);
    CHECK(candidate_wf.gpuInferenceEnabled());
    CHECK(candidate_wf.numericStorage().shared_gpu_bytes == storage.shared_gpu_bytes);
    CHECK(candidate_wf.numericStorage().shared_image_identity == storage.shared_image_identity);
    CHECK(metrics.snapshot().live_bytes == before.live_bytes + isolation.device_bytes);
    // Scratch preparation may transfer small descriptors; shared image identity and
    // the exact additional live bytes, not zero traffic, establish no initial P copy.
    const auto publication = candidate_wf.gpuParameterPublicationRequirements();
    const auto private_live = metrics.snapshot().live_bytes;
    const auto imaginary = std::find_if(initialized.layouts.begin(), initialized.layouts.end(),
        [](const auto& leaf) { return leaf.name == "w_imag"; });
    const auto real = std::find_if(initialized.layouts.begin(), initialized.layouts.end(),
        [](const auto& leaf) { return leaf.name == "w"; });
    REQUIRE(imaginary != initialized.layouts.end());
    REQUIRE(real != initialized.layouts.end());
    for (std::size_t step = 0; step < 2; ++step)
    {
      auto candidate = candidate_wf.snapshotParameters();
      candidate.values[real->begin] += .001 * (step + 1);
      candidate.values[imaginary->begin] += .002 * (step + 1);
      const auto before_prepare = metrics.snapshot();
      {
        auto discarded = candidate_wf.prepareBudgetedGpuParameterPublication(candidate, candidate.version,
            publication.device_bytes, publication.host_bytes);
        CHECK(metrics.snapshot().live_bytes == before_prepare.live_bytes + publication.device_bytes);
        CHECK(metrics.snapshot().allocation_calls == before_prepare.allocation_calls + 1);
        CHECK(metrics.snapshot().uploaded_bytes == before_prepare.uploaded_bytes + publication.device_bytes);
      }
      CHECK(metrics.snapshot().live_bytes == before_prepare.live_bytes);
      CHECK(candidate_wf.snapshotParameters().version == candidate.version);
      CHECK(candidate_wf.publishBudgetedGpuParameters(candidate, candidate.version,
          publication.device_bytes, publication.host_bytes) == candidate.version + 1);
      CHECK(cpu.publishParameters(candidate, candidate.version) == candidate.version + 1);
      CHECK(metrics.snapshot().live_bytes == private_live + publication.device_bytes);
      close(evaluate(candidate_wf, private_p), evaluate(cpu, cpu_p));
      for (int electron = 0; electron < p.getTotalNum(); ++electron)
      {
        close(private_p.L[electron], cpu_p.L[electron]);
        for (int d = 0; d < 3; ++d) close(private_p.G[electron][d], cpu_p.G[electron][d]);
      }
      close(candidate_wf.observeCoulombEnergy(private_p, -2.7).kinetic, cpu.observeCoulombEnergy(cpu_p, -2.7).kinetic);
      CHECK(score(*retained) == original_score);
      CHECK(gpu.snapshotParameters().values == original.values);
      CHECK(sibling.snapshotParameters().values == original.values);
      CHECK(gpu.snapshotParameters().version == original.version);
      CHECK(sibling.snapshotParameters().version == original.version);
      CHECK(evaluate(gpu, p) == original_log);
      CHECK(p.G == original_g); CHECK(p.L == original_l);
      CHECK(evaluate(sibling, sibling_p) == original_log);
      CHECK(sibling_p.G == original_g); CHECK(sibling_p.L == original_l);
      CHECK_THROWS(candidate_wf.publishBudgetedGpuParameters(candidate, candidate.version,
          publication.device_bytes, publication.host_bytes));
      auto invalid = candidate_wf.snapshotParameters();
      invalid.values[0] = std::numeric_limits<double>::quiet_NaN();
      CHECK_THROWS(candidate_wf.publishBudgetedGpuParameters(invalid, invalid.version,
          publication.device_bytes, publication.host_bytes));
      CHECK(candidate_wf.snapshotParameters().version == candidate.version + 1);
      CHECK(candidate_wf.snapshotParameters().values == cpu.snapshotParameters().values);
    }
  }
  CHECK(metrics.snapshot().live_bytes == outside.live_bytes);
}

TEST_CASE("PsiFormer explicit GPU parameter publication reuses inference storage atomically",
          "[wavefunction][psiformer][complex_gpu_publication]")
{
  using namespace wftrain;
  namespace dev = ps::device;
  int devices = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices)
  {
    WARN("NOT_RUN: GPU publication requires an available HIP device");
    return;
  }
  REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
  {
    WARN("NOT_RUN: GPU publication requires an available CUDA device");
    return;
  }
  REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
  ScopedComplexGpuControls controls;
  auto& metrics = dev::deviceMemoryMetrics(ordinal);
  const auto outside = metrics.snapshot();
  {
    const auto lattice = cell();
    const SimulationCell simulation_cell(lattice);
    auto p = electrons(simulation_cell), cpu_p = electrons(simulation_cell), clone_p = electrons(simulation_cell);
    auto source = ions(simulation_cell);
    const auto initialized = parameters();
    const std::array<double, 3> twist{.17, -.11, .07};
    ps::ExecutionEnvironment env;
    env.boundary = ps::BoundaryCondition::PERIODIC;
    env.geometry_feature_policy = ps::GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    env.periodic_axes = {true, true, true};
    env.head_representation = ps::HeadRepresentation::PAIRED_REAL_COMPLEX;
    env.amplitude_scalar_domain = ps::ScalarDomain::COMPLEX;
    for (int a = 0; a < 3; ++a)
      for (int d = 0; d < 3; ++d) env.lattice_vectors[a][d] = lattice.R(a, d);
    const auto plan = ps::PsiFormerExecutionPlan::fromParameters(initialized, initialized.model_shape, env);
    {
      // Numerical preparation must not revoke an already completed result, even
      // when a foreign family fails admission. Adoption revokes only at commit.
      std::vector<double> nuclei(3 * source.getTotalNum()), positions(3 * p.getTotalNum());
      for (int i = 0; i < source.getTotalNum(); ++i)
        for (int d = 0; d < 3; ++d) nuclei[3 * i + d] = source.R[i][d];
      for (int i = 0; i < p.getTotalNum(); ++i)
        for (int d = 0; d < 3; ++d) positions[3 * i + d] = p.R[i][d];
      auto image = dev::PreparedDeviceModel::preparePairedComplexValue(plan, initialized.values, nuclei);
      compute::Queue<PlatformKind::CUDA> queue;
      dev::AcceleratorBlasHandle blas(queue);
      auto value = dev::PreparedDeviceComplexValue::prepare(image, 1, queue, blas, 64 * 1024 * 1024);
      auto full = dev::PreparedDeviceComplexSpatial::prepare(image, 1, queue, blas, 64 * 1024 * 1024);
      value->evaluate(positions);
      full->evaluate(positions);
      const auto value_view = value->resultView();
      const auto full_view = full->resultView();
      auto changed = initialized.values;
      changed[0] += .002;
      auto successor = image->makeUpdatedImage(changed, 1);
      auto foreign = dev::PreparedDeviceModel::preparePairedComplexValue(plan, changed, nuclei, 1);
      CHECK_THROWS(value->validateModelRebind(foreign));
      CHECK_THROWS(full->validateModelRebind(foreign));
      value->validateModelRebind(successor);
      full->validateModelRebind(successor);
      std::vector<dev::ComplexValueResult> amplitude(1);
      std::vector<cuDoubleComplex> gradient(12), laplacian(4), ratio(4);
      CHECK_NOTHROW(value->download(value_view, amplitude));
      CHECK_NOTHROW(full->download(full_view, amplitude, gradient, laplacian, ratio));
      const auto before_adoption = metrics.snapshot();
      value->adoptPreparedModel(successor);
      full->adoptPreparedModel(successor);
      const auto after_adoption = metrics.snapshot();
      CHECK(after_adoption.allocation_calls == before_adoption.allocation_calls);
      CHECK(after_adoption.free_calls == before_adoption.free_calls);
      CHECK(after_adoption.upload_calls == before_adoption.upload_calls);
      CHECK(after_adoption.download_calls == before_adoption.download_calls);
      CHECK(after_adoption.device_copy_calls == before_adoption.device_copy_calls);
      CHECK_THROWS(value->download(value_view, amplitude));
      CHECK_THROWS(full->download(full_view, amplitude, gradient, laplacian, ratio));
      value->evaluate(positions);
      full->evaluate(positions);
      CHECK(value->resultView().data == value_view.data);
      CHECK(full->resultView().gradient == full_view.gradient);
    }
    std::size_t targets = 0;
    SECTION("FULL") {}
    SECTION("K3 tail") { targets = 3; }
    if (targets)
      controls.target(targets, dev::estimateComplexSpatialTargetCost(plan, targets).device_bytes,
          dev::estimateHeadFreeTargetPrefixCost(plan, targets).charged_scalar_work_units);
    PsiFormerComplexWF cpu("publication", initialized, cpu_p, source, clone_cap, twist);
    RuntimeOptions options;
    TrialWaveFunction trial(options);
    auto owner = std::make_unique<PsiFormerComplexWF>("publication", initialized, p, source, clone_cap,
                                                     twist, 64 * 1024 * 1024);
    auto& gpu = *owner;
    trial.addComponent(std::move(owner));
    const auto physical_identity = gpu.trainingConfigurationFingerprint();
    CHECK(physical_identity != cpu.trainingConfigurationFingerprint());
    auto clone_owner = gpu.makeClone(clone_p);
    auto& clone = dynamic_cast<PsiFormerComplexWF&>(*clone_owner);
    RefVectorWithLeader<TrialWaveFunction> trials(trial, {trial});
    RefVectorWithLeader<ParticleSet> particles(p, {p});
    const auto required = gpu.gpuParameterPublicationRequirements();
    const auto initial_storage = gpu.numericStorage();
    const auto parameter_bytes = sizeof(double) * gpu.parameterSchema().parameterCount();
    CHECK(required.device_bytes == parameter_bytes);
    CHECK(required.host_bytes == parameter_bytes);
    CHECK(metrics.snapshot().live_bytes - outside.live_bytes ==
        initial_storage.shared_gpu_bytes + initial_storage.shared_image_bytes);
    const auto* initial_image = static_cast<const dev::PreparedDeviceModel*>(initial_storage.shared_image_identity);
    const auto* metadata_identity = initial_image->metadataFamily().get();
    const auto* geometry_identity = initial_image->nuclei();
    CHECK_THROWS(cpu.gpuParameterPublicationRequirements());
    const auto refresh = [&]() {
      CHECK(gpu.trainingConfigurationFingerprint() == physical_identity);
      CHECK(clone.trainingConfigurationFingerprint() == physical_identity);
      close(evaluate(gpu, p), evaluate(cpu, cpu_p));
      close(evaluate(clone, clone_p), cpu.get_log_value());
      for (int e = 0; e < 4; ++e)
      {
        for (int d = 0; d < 3; ++d)
        {
          close(p.G[e][d], cpu_p.G[e][d]);
          close(clone_p.G[e][d], cpu_p.G[e][d]);
        }
        close(p.L[e], cpu_p.L[e]);
        close(clone_p.L[e], cpu_p.L[e]);
      }
      close(gpu.observeCoulombEnergy(p, -2.7).kinetic, cpu.observeCoulombEnergy(cpu_p, -2.7).kinetic);
    };
    refresh();
    const auto original = gpu.snapshotParameters();
    {
      auto discarded = original;
      discarded.values[0] += .017;
      const auto before_prepare = metrics.snapshot();
      auto prepared = gpu.prepareBudgetedGpuParameterPublication(discarded, original.version,
          required.device_bytes, required.host_bytes);
      CHECK(metrics.snapshot().live_bytes == before_prepare.live_bytes + required.device_bytes);
      CHECK(metrics.snapshot().allocation_calls == before_prepare.allocation_calls + 1);
      CHECK(metrics.snapshot().uploaded_bytes == before_prepare.uploaded_bytes + parameter_bytes);
      // A reader cannot observe old/new partial state while the action retains authority.
      auto reader = std::async(std::launch::async, [&] { return gpu.snapshotParameters(); });
      const bool excluded = reader.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout;
      prepared.reset();
      const auto after_discard = reader.get();
      CHECK(excluded);
      CHECK(after_discard.values == original.values);
      CHECK(after_discard.version == original.version);
      CHECK(metrics.snapshot().live_bytes == before_prepare.live_bytes);
      CHECK(gpu.numericStorage().shared_image_identity == initial_storage.shared_image_identity);
      CHECK_NOTHROW(gpu.observeCoulombEnergy(p, -2.7));
    }
    const auto reject = [&](const StructuredParameterSnapshot& bad, std::size_t version,
                            std::size_t device_cap, std::size_t host_cap) {
      const auto before = metrics.snapshot();
      CHECK_THROWS(gpu.publishBudgetedGpuParameters(bad, version, device_cap, host_cap));
      CHECK(metrics.snapshot().allocation_calls == before.allocation_calls);
      CHECK(gpu.snapshotParameters().values == original.values);
      CHECK(gpu.snapshotParameters().version == original.version);
      CHECK(gpu.numericStorage().shared_image_identity == initial_storage.shared_image_identity);
      CHECK_NOTHROW(gpu.observeCoulombEnergy(p, -2.7));
    };
    reject(original, original.version, required.device_bytes - 1, required.host_bytes);
    reject(original, original.version, required.device_bytes, required.host_bytes - 1);
    reject(original, original.version, 0, required.host_bytes);
    reject(original, original.version, required.device_bytes, std::numeric_limits<std::size_t>::max());
    auto bad = original;
    bad.schema_fingerprint += "bad";
    reject(bad, original.version, required.device_bytes, required.host_bytes);
    bad = original;
    bad.values.pop_back();
    reject(bad, original.version, required.device_bytes, required.host_bytes);
    bad = original;
    bad.version = original.version + 1;
    reject(bad, bad.version, required.device_bytes, required.host_bytes);
    bad = original;
    bad.values[0] = std::numeric_limits<double>::quiet_NaN();
    reject(bad, original.version, required.device_bytes, required.host_bytes);
    bad = original;
    bad.version = std::numeric_limits<std::size_t>::max();
    reject(bad, bad.version, required.device_bytes, required.host_bytes);
    clone_p.makeMove(0, ParticleSet::PosType{.01, 0, 0});
    clone.ratio(clone_p, 0);
    reject(original, original.version, required.device_bytes, required.host_bytes);
    clone.restore(0);
    clone_p.rejectMove(0);
    const auto saved_source = source.R[0][0];
    source.R[0][0] += .01;
    // Do not use observation while source differs; check native/image identity here.
    const auto before_source = metrics.snapshot();
    CHECK_THROWS(gpu.publishBudgetedGpuParameters(original, original.version, required.device_bytes, required.host_bytes));
    CHECK(metrics.snapshot().allocation_calls == before_source.allocation_calls);
    CHECK(gpu.snapshotParameters().values == original.values);
    CHECK(gpu.numericStorage().shared_image_identity == initial_storage.shared_image_identity);
    source.R[0][0] = saved_source;
    refresh();

    // With no derivative owner retaining the old image, the prepared action
    // alone must prevent DEVICE retirement until after commit unlocks authority.
    {
      auto candidate = original;
      candidate.values[0] += .0007;
      auto prepared = gpu.prepareBudgetedGpuParameterPublication(candidate, original.version,
          required.device_bytes, required.host_bytes);
      const auto before_commit = metrics.snapshot();
      CHECK(prepared->commit() == original.version + 1);
      const auto after_commit = metrics.snapshot();
      CHECK(after_commit.allocation_calls == before_commit.allocation_calls);
      CHECK(after_commit.free_calls == before_commit.free_calls);
      CHECK(after_commit.upload_calls == before_commit.upload_calls);
      CHECK(after_commit.download_calls == before_commit.download_calls);
      CHECK(after_commit.device_copy_calls == before_commit.device_copy_calls);
      CHECK(cpu.publishParameters(candidate, original.version) == original.version + 1);
      prepared.reset();
      CHECK(metrics.snapshot().live_bytes == after_commit.live_bytes - parameter_bytes);
      refresh();
    }

    const auto derivative = gpu.gpuDerivativeRequirements(1, 1, 3, 7);
    const auto make_stream = [&](std::size_t ordinal) {
      return gpu.makeBudgetedGpuDerivativeOperator(trials, particles, ordinal, 0, 1, 3, 7,
          derivative.device_bytes, derivative.host_bytes, derivative.matrix_work);
    };
    const auto apply = [&](StreamingDerivativeOperator& stream, ComplexStreamingSink& sink) {
      const std::vector<C> coefficient{{.3, -.2}};
      const auto& schema = stream.parameterSchema();
      const VJPCoefficientChannel channel{"score", DerivativeProduct::SCORE_VJP,
          {schema.providerId(), schema.fingerprint(), stream.parameterVersion(), stream.batchOrdinal(), 0,
           {coefficient.data(), 1}}, 0};
      stream.applyVJPs({&channel, 1}, DerivativeAdjoint::TRANSPOSE, sink);
    };
    auto old_stream = make_stream(1);
    ComplexStreamingSink old_output;
    apply(*old_stream, old_output);
    const auto completed_old_values = old_output.values;
    const auto single_generation = metrics.snapshot().live_bytes;
    const auto imaginary = std::find_if(initialized.layouts.begin(), initialized.layouts.end(),
        [](const auto& leaf) { return leaf.name == "w_imag"; });
    REQUIRE(imaginary != initialized.layouts.end());
    // Invalid process controls AFTER construction must not alter the frozen FULL/K3 profile.
    controls.set("PSIFORMER_GPU_VGL_TARGETS", "not-a-number");
    for (std::size_t step = 0; step < 2; ++step)
    {
      auto candidate = gpu.snapshotParameters();
      const auto expected_version = candidate.version;
      candidate.values[0] += .001 * (step + 1);
      candidate.values[imaginary->begin] += .002 * (step + 1);
      if (step == 0)
      {
        const auto before_prepare = metrics.snapshot();
        auto prepared = gpu.prepareBudgetedGpuParameterPublication(candidate, expected_version,
            required.device_bytes, required.host_bytes);
        const auto before_commit = metrics.snapshot();
        CHECK(before_commit.allocation_calls == before_prepare.allocation_calls + 1);
        CHECK(before_commit.uploaded_bytes == before_prepare.uploaded_bytes + parameter_bytes);
        CHECK(prepared->commit() == expected_version + 1);
        const auto after_commit = metrics.snapshot();
        CHECK(after_commit.allocation_calls == before_commit.allocation_calls);
        CHECK(after_commit.free_calls == before_commit.free_calls);
        CHECK(after_commit.upload_calls == before_commit.upload_calls);
        CHECK(after_commit.download_calls == before_commit.download_calls);
        CHECK(after_commit.device_copy_calls == before_commit.device_copy_calls);
      }
      else
        CHECK(gpu.publishBudgetedGpuParameters(candidate, expected_version,
            required.device_bytes, required.host_bytes) == expected_version + 1);
      CHECK(cpu.publishParameters(candidate, expected_version) == expected_version + 1);
      CHECK(gpu.snapshotParameters().version == expected_version + 1);
      CHECK(cpu.snapshotParameters().version == expected_version + 1);
      CHECK(gpu.snapshotParameters().values == cpu.snapshotParameters().values);
      const auto* image = static_cast<const dev::PreparedDeviceModel*>(gpu.numericStorage().shared_image_identity);
      CHECK(image->metadataFamily().get() == metadata_identity);
      CHECK(image->nuclei() == geometry_identity);
      CHECK(metrics.snapshot().live_bytes == single_generation + required.device_bytes);
      CHECK_THROWS(gpu.observeCoulombEnergy(p, -2.7));
      CHECK_THROWS(clone.observeCoulombEnergy(clone_p, -2.7));
      ComplexStreamingSink stale;
      const auto before_stale = metrics.snapshot();
      CHECK_THROWS(apply(*old_stream, stale));
      CHECK(metrics.snapshot().allocation_calls == before_stale.allocation_calls);
      CHECK(old_output.values == completed_old_values);
      refresh();
      auto next_stream = make_stream(step + 2);
      CHECK(next_stream->storageDiagnostics().shared_workspace_identity != old_stream->storageDiagnostics().shared_workspace_identity);
      CHECK(metrics.snapshot().live_bytes == single_generation + required.device_bytes + derivative.device_bytes);
      auto shared = make_stream(step + 9);
      CHECK(shared->storageDiagnostics().shared_workspace_identity == next_stream->storageDiagnostics().shared_workspace_identity);
      ComplexStreamingSink next_output;
      apply(*next_stream, next_output);
      RefVectorWithLeader<WaveFunctionComponent> cpu_components(cpu, {cpu});
      RefVectorWithLeader<ParticleSet> cpu_particles(cpu_p, {cpu_p});
      auto oracle = cpu.makeBudgetedStreamingDerivativeOperator(cpu_components, cpu_particles, step + 2, 0, 7,
          cpu.streamingDerivativeRequiredBytes(1));
      ComplexStreamingSink expected;
      apply(*oracle, expected);
      for (std::size_t parameter = 0; parameter < candidate.values.size(); ++parameter)
        close(next_output.values[0][parameter], expected.values[0][parameter], 3e-9);
      shared.reset();
      old_stream.reset();
      CHECK(metrics.snapshot().live_bytes == single_generation);
      old_stream = std::move(next_stream);
      CHECK(gpu.gpuParameterPublicationRequirements().device_bytes == required.device_bytes);
      CHECK(gpu.gpuParameterPublicationRequirements().host_bytes == required.host_bytes);
      CHECK(gpu.numericStorage().shared_gpu_bytes == initial_storage.shared_gpu_bytes);
      CHECK_THROWS(gpu.publishParameters(candidate, expected_version));
      CHECK(gpu.structuredParameterProvider() == nullptr);
      CHECK_THROWS(gpu.prepareBudgetedGpuParameterPublication(original, original.version,
          required.device_bytes, required.host_bytes));
    }
    std::printf("GPU publication K=%zu generations=3 new_device=%zu new_host=%zu derivative_device=%zu retirement_live=%zu\n",
        targets, required.device_bytes, required.host_bytes, derivative.device_bytes, single_generation);
  }
  CHECK(metrics.snapshot().live_bytes == outside.live_bytes);
}
#endif

#ifdef ENABLE_CUDA
#ifndef MIXED_PRECISION
TEST_CASE("PsiFormer paired B2 immediate replay owns every live tile and tail product",
          "[wavefunction][psiformer][complex_gpu_immediate_full_replay]")
{
  namespace dev = ps::device;
  int count = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
  REQUIRE(hipGetDeviceCount(&count) == hipSuccess);
  REQUIRE(count > 0);
  REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
  REQUIRE(count > 0);
  REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
  REQUIRE(dev::configuredDeviceMemoryAdmission().limit_bytes != 0);
  ScopedComplexGpuControls controls;
  controls.set("PSIFORMER_GPU_COMPLEX_TILE_CAPACITY", "2");
  controls.set("PSIFORMER_GPU_ACTIVE_INFERENCE_CROWDS", "1");
  auto& metrics = dev::deviceMemoryMetrics(ordinal);
  const auto outside = metrics.snapshot();
  {
    const SimulationCell simulation_cell(cell());
    auto p0 = electrons(simulation_cell), p1 = electrons(simulation_cell), p2 = electrons(simulation_cell);
    auto source = ions(simulation_cell);
    const std::array<double, 3> twist{.125, -.0625, .03125};
    std::array<ParticleSet*, 3> particles{&p0, &p1, &p2};
    for (auto* p : particles)
      p->setTwist(ParticleSet::PosType{twist[0], twist[1], twist[2]});
    p1.R[0][0] += .11;
    p2.R[3][1] -= .09;
    p1.update();
    p2.update();
    PsiFormerComplexWF gpu("replay_b2", parameters(), p0, source, clone_cap, twist,
                           64 * 1024 * 1024, {}, "pseudopotential");
    auto clone1 = gpu.makeClone(p1), clone2 = gpu.makeClone(p2);
    std::array<PsiFormerComplexWF*, 3> components{
        &gpu, dynamic_cast<PsiFormerComplexWF*>(clone1.get()), dynamic_cast<PsiFormerComplexWF*>(clone2.get())};
    REQUIRE(components[1] != nullptr);
    REQUIRE(components[2] != nullptr);
    RefVectorWithLeader<WaveFunctionComponent> team(gpu, {gpu, *clone1, *clone2});
    RefVectorWithLeader<ParticleSet> targets(p0, {p0, p1, p2});
    RefVector<ParticleSet::ParticleGradient> gradients{p0.G, p1.G, p2.G};
    RefVector<ParticleSet::ParticleLaplacian> laplacians{p0.L, p1.L, p2.L};
    ResourceCollection resource("paired B2 immediate replay");
    gpu.createResource(resource);
    using Value = WaveFunctionComponent::ValueType;
    std::array<std::unique_ptr<ParticleSet>, 3> saved;
    std::array<PsiFormerComplexWF::ImmediateFullReplayContext, 3> contexts;
    std::array<std::vector<Value>, 3> fresh_g, fresh_l;
    std::array<WaveFunctionComponent::LogValue, 3> logs;
    const auto matches = [&](std::size_t i) {
      return components[i]->canReuseImmediateFullSpatialReplay(*particles[i], *saved[i],
          fresh_g[i], fresh_l[i], contexts[i], std::real(logs[i]), std::imag(logs[i]));
    };
    const auto capture = [&] {
      for (std::size_t i = 0; i < particles.size(); ++i)
      {
        saved[i] = std::make_unique<ParticleSet>(*particles[i]);
        contexts[i] = components[i]->immediateFullReplayContext();
        logs[i] = components[i]->get_log_value();
        fresh_g[i].clear();
        fresh_l[i].clear();
        for (int e = 0; e < particles[i]->getTotalNum(); ++e)
        {
          for (int d = 0; d < 3; ++d)
            fresh_g[i].push_back(particles[i]->G[e][d]);
          fresh_l[i].push_back(particles[i]->L[e]);
        }
        REQUIRE(matches(i));
      }
    };

    // Three walkers exercise a full B2 tile and its live one-walker tail.
    for (int acquisition = 0; acquisition < 2; ++acquisition)
    {
      ResourceCollectionTeamLock<WaveFunctionComponent> lease(resource, team);
      for (auto* p : particles) { p->G = 0; p->L = 0; }
      gpu.mw_evaluateLog(team, targets, gradients, laplacians);
      capture();
      const auto warm = metrics.snapshot();
      for (std::size_t i = 0; i < particles.size(); ++i)
      {
        CHECK(matches(i));
        CHECK(matches(i)); // The component proof is read-only, not the driver's one-use receipt.
      }
      CHECK(metrics.snapshot().allocation_calls == warm.allocation_calls);
      CHECK(metrics.snapshot().free_calls == warm.free_calls);

      // Overwrite the shared lane with walker0 without invalidating copied walker1/tail products.
      p0.G = 0;
      p0.L = 0;
      gpu.evaluateLog(p0, p0.G, p0.L);
      CHECK(matches(1));
      CHECK(matches(2));
      const auto coordinate = p2.R[0][0];
      p2.R[0][0] = std::numeric_limits<double>::quiet_NaN();
      CHECK_FALSE(matches(2));
      CHECK_THROWS(gpu.mw_evaluateLog(team, targets, gradients, laplacians));
      p2.R[0][0] = coordinate;
      CHECK(matches(1));
      CHECK(matches(2)); // Failed whole-team preparation cannot partially publish any clone.
      CHECK(metrics.snapshot().allocation_calls == warm.allocation_calls);
      CHECK(metrics.snapshot().free_calls == warm.free_calls);
    }

    // A frozen B2 family remains B2; new image/version invalidates all old clone proofs.
    const auto publication = gpu.gpuParameterPublicationRequirements();
    auto candidate = gpu.snapshotParameters();
    candidate.values[0] += .001;
    controls.set("PSIFORMER_GPU_COMPLEX_TILE_CAPACITY", "4");
    REQUIRE(gpu.publishBudgetedGpuParameters(candidate, candidate.version,
        publication.device_bytes, publication.host_bytes) == candidate.version + 1);
    for (std::size_t i = 0; i < particles.size(); ++i)
      CHECK_FALSE(matches(i));
    {
      ResourceCollectionTeamLock<WaveFunctionComponent> lease(resource, team);
      for (auto* p : particles) { p->G = 0; p->L = 0; }
      gpu.mw_evaluateLog(team, targets, gradients, laplacians);
      capture();
    }
  }
  CHECK(metrics.snapshot().live_bytes == outside.live_bytes);
  CHECK(metrics.snapshot().failed_frees == outside.failed_frees);
}
#endif

TEST_CASE("PsiFormer paired GPU B1 retains complete collective products across walker tiles",
          "[wavefunction][psiformer][complex_gpu_collective_b1]")
{
  const auto selected = GENERATE(1, 2, 3, 4);
  CAPTURE(selected);
  ScopedComplexGpuControls controls;
  controls.set("PSIFORMER_GPU_COMPLEX_TILE_CAPACITY", "1");
  controls.set("PSIFORMER_GPU_ACTIVE_INFERENCE_CROWDS", "1");
  for (const auto twist : {std::array<double, 3>{}, std::array<double, 3>{.17, -.11, .07}})
  {
    const SimulationCell simulation_cell(cell());
    auto p0 = electrons(simulation_cell), p1 = electrons(simulation_cell);
    p1.R[0][1] += .03;
    p1.update();
    auto source = ions(simulation_cell);
    PsiFormerComplexWF gpu("b1", parameters(), p0, source, clone_cap, twist, 64 * 1024 * 1024);
    auto other = gpu.makeClone(p1);
    auto& clone = dynamic_cast<PsiFormerComplexWF&>(*other);
    REQUIRE(gpu.supportsMultiParticleMoves());
    const std::array<C, 2> initial{evaluate(gpu, p0), evaluate(clone, p1)};
    RefVectorWithLeader<WaveFunctionComponent> team(gpu, {gpu, clone});
    RefVectorWithLeader<ParticleSet> targets(p0, {p0, p1});
    const std::array<decltype(p0.R), 2> old_positions{p0.R, p1.R};
    std::vector<ParticleSet::IndexType> indices;
    std::vector<ParticleSet::PosType> positions;
    for (std::size_t i = 0; i < 2; ++i)
      for (int row = 0; row < selected; ++row)
      {
        const int electron = selected == 4 ? row : row + 1;
        indices.push_back(electron);
        positions.push_back(targets[i].R[electron] + ParticleSet::PosType{.002 * (i + 1), -.003, .001});
      }
    const MCMultiParticleMoves<CoordsType::POS> moves(
        {0, std::size_t(selected), std::size_t(2 * selected)}, indices, positions);
    ParticleSet::ParticleGradient g0(4), g1(4);
    ParticleSet::ParticleLaplacian l0(4), l1(4);
    RefVector<ParticleSet::ParticleGradient> gs{g0, g1};
    RefVector<ParticleSet::ParticleLaplacian> ls{l0, l1};
    std::vector<C> ratios(2);
    ResourceCollection resources("B1 collective");
    gpu.createResource(resources);
    ResourceCollectionTeamLock<WaveFunctionComponent> lease(resources, team);
    for (int repeat = 0; repeat < 2; ++repeat)
    {
      g0 = g1 = C{};
      l0 = l1 = C{};
      gpu.mw_evaluateMultiParticleMove(team, targets, moves, ratios, gs, ls);
      for (std::size_t i = 0; i < 2; ++i)
      {
        auto proposed = electrons(simulation_cell);
        proposed.R = old_positions[i];
        for (int row = 0; row < selected; ++row)
          proposed.R[indices[selected * i + row]] = positions[selected * i + row];
        proposed.update();
        PsiFormerComplexWF oracle("oracle", parameters(), proposed, source, clone_cap, twist);
        close(std::exp(ratios[i]), std::exp(evaluate(oracle, proposed) - initial[i]), 2e-9);
        for (std::size_t e = 0; e < 4; ++e)
        {
          close(ls[i].get()[e], proposed.L[e], 2e-8);
          for (int d = 0; d < 3; ++d)
            close(gs[i].get()[e][d], proposed.G[e][d], 2e-8);
        }
      }
      CHECK_THROWS(gpu.mw_accept_rejectMultiParticleMove(team, targets, moves, std::vector<bool>{false}));
      gpu.mw_accept_rejectMultiParticleMove(team, targets, moves, std::vector<bool>{false, false});
      CHECK(gpu.get_log_value() == initial[0]);
      CHECK(clone.get_log_value() == initial[1]);
    }
    g0 = g1 = C{}; l0 = l1 = C{};
    gpu.mw_evaluateMultiParticleMove(team, targets, moves, ratios, gs, ls);
    const auto accepted_log = initial[0] + ratios[0];
    for (int row = 0; row < selected; ++row)
      p0.R[indices[row]] = positions[row];
    p0.update();
    gpu.mw_accept_rejectMultiParticleMove(team, targets, moves, std::vector<bool>{true, false});
    close(gpu.get_log_value(), accepted_log, 2e-12);
    CHECK(clone.get_log_value() == initial[1]);
    for (int electron = 0; electron < 4; ++electron)
      for (int d = 0; d < 3; ++d)
        CHECK(p1.R[electron][d] == old_positions[1][electron][d]);
    for (int row = selected; row < 4; ++row)
      CHECK(p0.R[(row + 1) % 4] == old_positions[0][(row + 1) % 4]);
    CHECK_NOTHROW(gpu.observeCoulombEnergy(p0, 0));
  }
}

TEST_CASE("PsiFormer dense paired GPU native tiles preserve phase tails and cache transactions",
          "[wavefunction][psiformer][complex_gpu_dense]")
{
  namespace dev = ps::device;
  int devices = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices)
  {
    WARN("NOT_RUN: paired dense inference requires an available HIP device");
    return;
  }
  REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
  {
    WARN("NOT_RUN: paired dense inference requires an available CUDA device");
    return;
  }
  REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
  if (!dev::configuredDeviceMemoryAdmission().limit_bytes)
  {
    WARN("NOT_RUN: paired dense inference test requires explicit native aggregate device admission");
    return;
  }
  ScopedComplexGpuControls controls;
  std::size_t target = 0;
  SECTION("FULL") {}
  SECTION("explicit K3") { target = 3; }
  controls.set("PSIFORMER_GPU_COMPLEX_TILE_CAPACITY", "2");
  auto& metrics = dev::deviceMemoryMetrics(ordinal);
  const auto outside = metrics.snapshot();
  const auto lattice = cell();
  const SimulationCell simulation_cell(lattice);
  for (const auto twist : {std::array<double, 3>{}, std::array<double, 3>{.17, -.11, .07}})
  {
    const auto cartesian_twist = lattice.k_cart(ParticleSet::PosType{twist[0], twist[1], twist[2]});
    auto p0 = electrons(simulation_cell, 3, 1), p1 = electrons(simulation_cell, 3, 1), p2 = electrons(simulation_cell, 3, 1);
    auto source = ions(simulation_cell);
    p1.R[0][0] += .11;
    p2.R[3][1] -= .09;
    p1.update();
    p2.update();
    auto initialized = parameters();
    initialized.model_shape.spin_up_electrons = 3;
    initialized.model_shape.spin_down_electrons = 1;
    ps::ExecutionEnvironment environment;
    environment.boundary = ps::BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = ps::GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    environment.periodic_axes = {true, true, true};
    environment.head_representation = ps::HeadRepresentation::PAIRED_REAL_COMPLEX;
    environment.amplitude_scalar_domain = ps::ScalarDomain::COMPLEX;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        environment.lattice_vectors[i][j] = simulation_cell.getLattice().R(i, j);
    const auto plan = ps::PsiFormerExecutionPlan::fromParameters(initialized, initialized.model_shape, environment);
    const auto spatial = target ? dev::estimateComplexSpatialTargetCost(plan, target, 2)
                                : dev::estimateComplexSpatialCost(plan, 2);
    const auto owner_bytes = dev::estimateComplexValueCost(plan, 2).device_bytes + spatial.device_bytes;
    if (target)
      controls.target(target, spatial.device_bytes,
          dev::estimateHeadFreeTargetPrefixCost(plan, target, 2).charged_scalar_work_units);
    const auto before_rejection = metrics.snapshot();
    CHECK_THROWS_AS(PsiFormerComplexWF("pf", initialized, p0, source, clone_cap, twist, owner_bytes - 1), std::length_error);
    CHECK(metrics.snapshot().allocation_calls == before_rejection.allocation_calls);
    controls.target(1, owner_bytes, 1);
    CHECK_THROWS_AS(PsiFormerComplexWF("pf", initialized, p0, source, clone_cap, twist, owner_bytes), std::length_error);
    for (const auto* name : {"PSIFORMER_GPU_VGL_TARGETS", "PSIFORMER_GPU_TARGET_WORKSPACE_BYTES", "PSIFORMER_GPU_TARGET_SCALAR_WORK_BUDGET"})
      ::unsetenv(name);

    if (target)
      controls.target(target, spatial.device_bytes,
          dev::estimateHeadFreeTargetPrefixCost(plan, target, 2).charged_scalar_work_units);

    PsiFormerComplexWF gpu("pf", initialized, p0, source, clone_cap, twist, owner_bytes);
    PsiFormerComplexWF cpu("pf", initialized, p0, source, clone_cap, twist);
    CHECK(gpu.supportsMultiParticleMoves());
    CHECK_FALSE(cpu.supportsMultiParticleMoves());
    auto gpu1 = gpu.makeClone(p1), gpu2 = gpu.makeClone(p2);
    auto cpu1 = cpu.makeClone(p1), cpu2 = cpu.makeClone(p2);
    std::array<ParticleSet*, 3> particles{&p0, &p1, &p2};
    std::array<WaveFunctionComponent*, 3> references{&cpu, cpu1.get(), cpu2.get()};
    std::array<WaveFunctionComponent*, 3> components{&gpu, gpu1.get(), gpu2.get()};
    RefVectorWithLeader<WaveFunctionComponent> team(gpu, {gpu, *gpu1, *gpu2});
    RefVectorWithLeader<ParticleSet> targets(p0, {p0, p1, p2});
    RefVector<ParticleSet::ParticleGradient> gradients{p0.G, p1.G, p2.G};
    RefVector<ParticleSet::ParticleLaplacian> laplacians{p0.L, p1.L, p2.L};
    ResourceCollection resource("paired dense test");
    gpu.createResource(resource);
    ResourceCollectionTeamLock<WaveFunctionComponent> lease(resource, team);
    using Grad = WaveFunctionComponent::GradType;
    // Accepted gradients require no accepted-amplitude cache and must not populate one.
    std::vector<Grad> accepted_gradients(3, Grad(C{9, 7}));
    const std::array<C, 3> uncached_logs{
        components[0]->get_log_value(), components[1]->get_log_value(), components[2]->get_log_value()};
    gpu.mw_evalGrad(team, targets, 0, accepted_gradients);
    for (std::size_t i = 0; i < 3; ++i)
    {
      CHECK(components[i]->get_log_value() == uncached_logs[i]);
      CHECK_THROWS(static_cast<PsiFormerComplexWF*>(components[i])->observeCoulombEnergy(*particles[i], 0));
    }
    const auto refresh = [&] {
      for (auto* p : particles) { p->G = 0; p->L = 0; }
      gpu.mw_evaluateGL(team, targets, gradients, laplacians, true);
      double learned_phase_gradient = 0;
      for (std::size_t walker = 0; walker < 3; ++walker)
      {
        ParticleSet::ParticleGradient expected_g(4);
        ParticleSet::ParticleLaplacian expected_l(4);
        expected_g = 0;
        expected_l = 0;
        const auto expected = references[walker]->evaluateLog(*particles[walker], expected_g, expected_l);
        const auto actual = components[walker]->get_log_value();
        CHECK(actual.real() == Catch::Approx(expected.real()).margin(2e-9));
        close(std::exp(C{0, actual.imag()}), std::exp(C{0, expected.imag()}), 2e-9);
        for (std::size_t e = 0; e < 4; ++e)
        {
          close(particles[walker]->L[e], expected_l[e], 2e-8);
          for (int d = 0; d < 3; ++d)
          {
            close(particles[walker]->G[e][d], expected_g[e][d], 2e-8);
            learned_phase_gradient = std::max(learned_phase_gradient,
                std::abs(expected_g[e][d].imag() - cartesian_twist[d]));
          }
        }
      }
      CHECK(learned_phase_gradient > 1e-6);
    };
    refresh();
    for (std::size_t i = 0; i < 3; ++i)
      for (int d = 0; d < 3; ++d)
        close(accepted_gradients[i][d], particles[i]->G[0][d], 2e-8);
    const auto warm = metrics.snapshot();
    refresh();
    CHECK(metrics.snapshot().allocation_calls == warm.allocation_calls);
    CHECK(metrics.snapshot().free_calls == warm.free_calls);
    accepted_gradients.assign(3, Grad(C{9, 7}));
    gpu.mw_evalGrad(team, targets, 3, accepted_gradients);
    CHECK(metrics.snapshot().allocation_calls == warm.allocation_calls);
    CHECK(metrics.snapshot().free_calls == warm.free_calls);
    for (std::size_t i = 0; i < 3; ++i)
      for (int d = 0; d < 3; ++d)
        close(accepted_gradients[i][d], particles[i]->G[3][d], 2e-8); // WRITE, not additive
    CHECK_THROWS_AS(gpu.mw_evalGrad(team, targets, -1, accepted_gradients), std::invalid_argument);
    CHECK_THROWS_AS(gpu.mw_evalGrad(team, targets, 4, accepted_gradients), std::invalid_argument);
    std::vector<Grad> wrong_accepted_extent(2);
    CHECK_THROWS_AS(gpu.mw_evalGrad(team, targets, 0, wrong_accepted_extent), std::invalid_argument);
    const auto scalar_gradient = gpu.evalGrad(p0, 0);
    for (int d = 0; d < 3; ++d)
      close(scalar_gradient[d], p0.G[0][d], 2e-8);
    // One-crowd dense inference now admits independent serialized products.
    CHECK_NOTHROW(gpu.gpuDerivativeRequirements(1, 1, 1, 1));

    auto aliases = gradients;
    aliases[2] = std::ref(p0.G);
    CHECK_THROWS_AS(gpu.mw_evaluateLog(team, targets, aliases, laplacians), std::invalid_argument);
    RefVectorWithLeader<WaveFunctionComponent> wrong_order(gpu, {gpu, *gpu2, *gpu1});
    CHECK_THROWS_AS(gpu.mw_evaluateLog(wrong_order, targets, gradients, laplacians), std::invalid_argument);
    std::array<C, 3> previous_logs;
    std::array<ParticleSet::ParticleGradient, 3> previous_g{p0.G, p1.G, p2.G};
    std::array<ParticleSet::ParticleLaplacian, 3> previous_l{p0.L, p1.L, p2.L};
    for (std::size_t i = 0; i < 3; ++i)
      previous_logs[i] = components[i]->get_log_value();
    const auto saved_position = p2.R[1][0];
    p2.R[1][0] = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS(gpu.mw_evaluateLog(team, targets, gradients, laplacians));
    p2.R[1][0] = saved_position;
    for (std::size_t walker = 0; walker < 3; ++walker)
    {
      CHECK(components[walker]->get_log_value() == previous_logs[walker]);
      CHECK_NOTHROW(static_cast<PsiFormerComplexWF*>(components[walker])->observeCoulombEnergy(*particles[walker], 0));
      for (std::size_t e = 0; e < 4; ++e)
      {
        CHECK(particles[walker]->L[e] == previous_l[walker][e]);
        for (int d = 0; d < 3; ++d)
          CHECK(particles[walker]->G[e][d] == previous_g[walker][e][d]);
      }
    }
    std::array<C, 3> expected_ratios;
    for (std::size_t walker = 0; walker < 3; ++walker)
    {
      particles[walker]->makeMove(0, ParticleSet::PosType{.013, -.007, .011});
      expected_ratios[walker] = references[walker]->ratio(*particles[walker], 0);
    }
    std::vector<C> ratios(3, C{9, 7});
    gpu.mw_calcRatio(team, targets, 0, ratios);
    for (std::size_t i = 0; i < 3; ++i)
      close(ratios[i], expected_ratios[i], 2e-9);
    ratios.assign(3, C{9, 7});
    p2.R[1][0] = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS(gpu.mw_calcRatio(team, targets, 0, ratios));
    p2.R[1][0] = saved_position;
    for (const auto ratio : ratios)
      CHECK(ratio == (C{9, 7}));
    for (std::size_t walker = 0; walker < 3; ++walker)
      if (walker != 1)
      {
        references[walker]->acceptMove(*particles[walker], 0);
        components[walker]->acceptMove(*particles[walker], 0);
        particles[walker]->acceptMove(0);
      }
      else
      {
        references[walker]->restore(0);
        components[walker]->restore(0);
        particles[walker]->rejectMove(0);
      }
    refresh();
    // Proposed FULL gradients batch through the same T2 owner, including the tail.
    // A separate CPU clone at the complete proposed configuration is the oracle.
    const C additive{.21, -.13};
    std::vector<Grad> proposed_gradients(3, Grad(additive));
    std::array<Grad, 3> expected_gradients;
    for (std::size_t walker = 0; walker < 3; ++walker)
    {
      particles[walker]->makeMove(3, ParticleSet::PosType{-.009, .012, -.005});
      expected_gradients[walker] = Grad(additive);
      expected_ratios[walker] = references[walker]->ratioGrad(*particles[walker], 3, expected_gradients[walker]);
      auto proposed_particles = electrons(simulation_cell, 3, 1);
      proposed_particles.R = particles[walker]->R;
      proposed_particles.R[3] = particles[walker]->activeR(3);
      proposed_particles.update();
      auto oracle = references[walker]->makeClone(proposed_particles);
      evaluate(*oracle, proposed_particles);
      for (int d = 0; d < 3; ++d)
        close(expected_gradients[walker][d], proposed_particles.G[3][d] + additive, 2e-8);
    }
    ratios.assign(3, C{9, 7});
    const auto before_ratio_grad = metrics.snapshot();
    gpu.mw_ratioGrad(team, targets, 3, ratios, proposed_gradients);
    CHECK(metrics.snapshot().allocation_calls == before_ratio_grad.allocation_calls);
    CHECK(metrics.snapshot().free_calls == before_ratio_grad.free_calls);
    for (std::size_t i = 0; i < 3; ++i)
    {
      close(ratios[i], expected_ratios[i], 2e-9);
      for (int d = 0; d < 3; ++d)
        close(proposed_gradients[i][d], expected_gradients[i][d], 2e-8);
    }
    // Even with active proposals, accepted gradients use old R and preserve all cache/P.G/L state.
    const std::array<ParticleSet::ParticleGradient, 3> accepted_g{p0.G, p1.G, p2.G};
    const std::array<ParticleSet::ParticleLaplacian, 3> accepted_l{p0.L, p1.L, p2.L};
    const std::array<C, 3> accepted_logs{
        components[0]->get_log_value(), components[1]->get_log_value(), components[2]->get_log_value()};
    accepted_gradients.assign(3, Grad(C{9, 7}));
    gpu.mw_evalGrad(team, targets, 3, accepted_gradients);
    for (std::size_t i = 0; i < 3; ++i)
    {
      CHECK(components[i]->get_log_value() == accepted_logs[i]);
      for (int d = 0; d < 3; ++d)
        close(accepted_gradients[i][d], accepted_g[i][3][d], 2e-8);
      for (std::size_t e = 0; e < 4; ++e)
      {
        CHECK(particles[i]->L[e] == accepted_l[i][e]);
        for (int d = 0; d < 3; ++d)
          CHECK(particles[i]->G[e][d] == accepted_g[i][e][d]);
      }
    }
    ratios.assign(3, C{9, 7});
    proposed_gradients.assign(3, Grad(additive));
    const std::array<C, 3> before_failure_logs{
        components[0]->get_log_value(), components[1]->get_log_value(), components[2]->get_log_value()};
    p2.R[1][0] = std::numeric_limits<double>::quiet_NaN();
    accepted_gradients.assign(3, Grad(C{9, 7}));
    CHECK_THROWS(gpu.mw_evalGrad(team, targets, 3, accepted_gradients));
    for (const auto& gradient : accepted_gradients)
      for (int d = 0; d < 3; ++d)
        CHECK(gradient[d] == (C{9, 7}));
    CHECK_THROWS(gpu.mw_ratioGrad(team, targets, 3, ratios, proposed_gradients));
    p2.R[1][0] = saved_position;
    for (std::size_t i = 0; i < 3; ++i)
    {
      CHECK(ratios[i] == (C{9, 7}));
      CHECK(components[i]->get_log_value() == before_failure_logs[i]);
      for (int d = 0; d < 3; ++d)
        CHECK(proposed_gradients[i][d] == additive);
    }
    // A late caller-output validation failure must also preserve prior proposals.
    proposed_gradients[2][2] = C{std::numeric_limits<double>::infinity(), 0};
    const auto invalid_gradients = proposed_gradients;
    CHECK_THROWS_AS(gpu.mw_ratioGrad(team, targets, 3, ratios, proposed_gradients), std::overflow_error);
    for (std::size_t i = 0; i < 3; ++i)
      for (int d = 0; d < 3; ++d)
        CHECK(proposed_gradients[i][d] == invalid_gradients[i][d]);
    for (const auto ratio : ratios)
      CHECK(ratio == (C{9, 7}));
    std::vector<Grad> wrong_extent(2);
    CHECK_THROWS_AS(gpu.mw_ratioGrad(team, targets, 3, ratios, wrong_extent), std::invalid_argument);
    for (std::size_t walker = 0; walker < 3; ++walker)
      if (walker != 1)
      {
        references[walker]->acceptMove(*particles[walker], 3);
        components[walker]->acceptMove(*particles[walker], 3);
        particles[walker]->acceptMove(3);
        CHECK_THROWS(static_cast<PsiFormerComplexWF*>(components[walker])->observeCoulombEnergy(*particles[walker], 0));
      }
      else
      {
        references[walker]->restore(3);
        components[walker]->restore(3);
        particles[walker]->rejectMove(3);
        CHECK_NOTHROW(static_cast<PsiFormerComplexWF*>(components[walker])->observeCoulombEnergy(*particles[walker], 0));
      }
    refresh();
    // Collective descriptors mix a selected slice, an empty identity slice and a full-system tail.
    using Moves = MCMultiParticleMoves<CoordsType::POS>;
    std::vector<ParticleSet::IndexType> selected_indices{0, 3, 0, 1, 2, 3};
    std::vector<ParticleSet::PosType> selected_positions;
    for (std::size_t i = 0; i < selected_indices.size(); ++i)
      selected_positions.push_back(particles[i < 2 ? 0 : 2]->R[selected_indices[i]] +
          ParticleSet::PosType{.004 * (i + 1), -.003, .002});
    const Moves moves({0, 2, 2, 6}, selected_indices, selected_positions);
    std::array<ParticleSet::ParticleGradient, 3> proposed_g{ParticleSet::ParticleGradient(4),
        ParticleSet::ParticleGradient(4), ParticleSet::ParticleGradient(4)};
    std::array<ParticleSet::ParticleLaplacian, 3> proposed_l{ParticleSet::ParticleLaplacian(4),
        ParticleSet::ParticleLaplacian(4), ParticleSet::ParticleLaplacian(4)};
    RefVector<ParticleSet::ParticleGradient> collective_g{proposed_g[0], proposed_g[1], proposed_g[2]};
    RefVector<ParticleSet::ParticleLaplacian> collective_l{proposed_l[0], proposed_l[1], proposed_l[2]};
    std::array<ParticleSet::ParticleGradient, 3> oracle_g = proposed_g;
    std::array<ParticleSet::ParticleLaplacian, 3> oracle_l = proposed_l;
    std::array<C, 3> oracle_logs, oracle_kinetic, initial_logs;
    const std::array<decltype(p0.R), 3> initial_positions{p0.R, p1.R, p2.R};
    for (std::size_t walker = 0; walker < 3; ++walker)
    {
      initial_logs[walker] = components[walker]->get_log_value();
      auto proposed = electrons(simulation_cell, 3, 1);
      proposed.R = particles[walker]->R;
      const auto slice = moves.slice(walker);
      for (std::size_t selected = 0; selected < slice.size(); ++selected)
        proposed.R[slice.particleIndex(selected)] = slice.proposedPosition(selected);
      proposed.update();
      auto oracle = references[walker]->makeClone(proposed);
      oracle_logs[walker] = evaluate(*oracle, proposed);
      oracle_g[walker] = proposed.G;
      oracle_l[walker] = proposed.L;
      oracle_kinetic[walker] = static_cast<PsiFormerComplexWF&>(*oracle).observeCoulombEnergy(proposed, 0).kinetic;
    }
    std::vector<C> log_ratios(3, C{9, 7});
    const auto initialize_collective_outputs = [&] {
      log_ratios.assign(3, C{9, 7});
      for (std::size_t i = 0; i < 3; ++i) { proposed_g[i] = additive; proposed_l[i] = additive; }
    };
    const auto collective_before = metrics.snapshot();
    initialize_collective_outputs();
    auto accepted_alias = collective_g;
    accepted_alias[2] = std::ref(p0.G);
    CHECK_THROWS_AS(gpu.mw_evaluateMultiParticleMove(team, targets, moves, log_ratios, accepted_alias, collective_l),
                    std::invalid_argument);
    gpu.mw_evaluateMultiParticleMove(team, targets, moves, log_ratios, collective_g, collective_l);
    for (std::size_t i = 0; i < 3; ++i)
    {
      close(std::exp(log_ratios[i]), std::exp(oracle_logs[i] - initial_logs[i]), 2e-9);
      CHECK(components[i]->get_log_value() == initial_logs[i]);
      for (std::size_t e = 0; e < 4; ++e)
      {
        close(proposed_l[i][e], oracle_l[i][e] + additive, 2e-8);
        for (int d = 0; d < 3; ++d)
          close(proposed_g[i][e][d], oracle_g[i][e][d] + additive, 2e-8);
      }
    }
    close(std::exp(log_ratios[1]), C{1}, 2e-9); // empty slice is a valid identity proposal
    const std::vector<bool> reject_all(3, false);
    CHECK_NOTHROW(gpu.mw_accept_rejectMultiParticleMove(team, targets, moves, reject_all));
    CHECK_THROWS(gpu.mw_accept_rejectMultiParticleMove(team, targets, moves, reject_all)); // exactly once
    for (std::size_t i = 0; i < 3; ++i)
      CHECK_NOTHROW(static_cast<PsiFormerComplexWF*>(components[i])->observeCoulombEnergy(*particles[i], 0));
    // A late additive-output failure publishes neither outputs nor pending state.
    initialize_collective_outputs();
    proposed_l[2][3] = C{std::numeric_limits<double>::infinity(), 0};
    CHECK_THROWS_AS(gpu.mw_evaluateMultiParticleMove(team, targets, moves, log_ratios, collective_g, collective_l),
                    std::overflow_error);
    for (std::size_t i = 0; i < 3; ++i)
    {
      CHECK(log_ratios[i] == (C{9, 7}));
      for (std::size_t e = 0; e < 4; ++e)
      {
        if (i != 2 || e != 3) CHECK(proposed_l[i][e] == additive);
        for (int d = 0; d < 3; ++d) CHECK(proposed_g[i][e][d] == additive);
      }
    }
    initialize_collective_outputs();
    gpu.mw_evaluateMultiParticleMove(team, targets, moves, log_ratios, collective_g, collective_l);
    WaveFunctionComponent::WFBufferType collective_buffer;
    CHECK_THROWS(gpu.registerData(p0, collective_buffer));
    CHECK_THROWS(gpu.updateBuffer(p0, collective_buffer, true));
    CHECK_THROWS(gpu.copyFromBuffer(p0, collective_buffer));
    CHECK_THROWS(gpu.ratio(p0, 0));
    CHECK_THROWS(gpu.ratioGrad(p0, 0, accepted_gradients[0]));
    CHECK_THROWS(gpu.evalGrad(p0, 0));
    CHECK_THROWS(gpu.acceptMove(p0, 0));
    CHECK_THROWS(gpu.restore(0));
    CHECK_THROWS(gpu.evaluateLog(p0, p0.G, p0.L));
    CHECK_THROWS(gpu.mw_evaluateLog(team, targets, gradients, laplacians));
    CHECK_THROWS(gpu.mw_evalGrad(team, targets, 0, accepted_gradients));
    CHECK_THROWS(gpu.mw_calcRatio(team, targets, 0, ratios));
    CHECK_THROWS(gpu.mw_ratioGrad(team, targets, 0, ratios, proposed_gradients));
    CHECK_THROWS(gpu.mw_evaluateMultiParticleMove(team, targets, moves, log_ratios, collective_g, collective_l));
    auto pending_snapshot = gpu.snapshotParameters();
    const auto pending_requirements = gpu.gpuParameterPublicationRequirements();
    CHECK_THROWS(gpu.publishBudgetedGpuParameters(pending_snapshot, pending_snapshot.version,
        pending_requirements.device_bytes, pending_requirements.host_bytes));
    auto wrong_positions = selected_positions;
    wrong_positions[0][0] += .001;
    const Moves wrong_moves({0, 2, 2, 6}, selected_indices, wrong_positions);
    CHECK_THROWS(gpu.mw_accept_rejectMultiParticleMove(team, targets, wrong_moves, reject_all));
    CHECK_THROWS(gpu.mw_accept_rejectMultiParticleMove(team, targets, moves, std::vector<bool>(2, false)));
    CHECK_THROWS(gpu.mw_accept_rejectMultiParticleMove(team, targets, moves, std::vector<bool>(3, true)));
    ResourceCollection wrong_resource("foreign collective loan");
    CHECK_THROWS(gpu.releaseResource(wrong_resource, team));
    for (std::size_t walker = 0; walker < 3; ++walker)
    {
      const auto slice = moves.slice(walker);
      for (std::size_t selected = 0; selected < slice.size(); ++selected)
        particles[walker]->R[slice.particleIndex(selected)] = slice.proposedPosition(selected);
      particles[walker]->update(); // caller installs actual absolute proposals
    }
    const std::vector<bool> accept_mask{true, false, true};
    p2.R[0][0] += .001;
    CHECK_THROWS(gpu.mw_accept_rejectMultiParticleMove(team, targets, moves, accept_mask));
    CHECK(components[0]->get_log_value() == initial_logs[0]); // no partial first-lane commit
    // Restore exact bits, rather than relying on floating-point add/subtract cancellation.
    p2.R[0] = selected_positions[2];
    gpu.mw_accept_rejectMultiParticleMove(team, targets, moves, accept_mask);
    for (std::size_t i = 0; i < 3; ++i)
    {
      if (!accept_mask[i]) { particles[i]->R = initial_positions[i]; particles[i]->update(); }
      CHECK(components[i]->get_log_value().real() == Catch::Approx(oracle_logs[i].real()).margin(2e-9));
      close(std::exp(C{0, components[i]->get_log_value().imag()}), std::exp(C{0, oracle_logs[i].imag()}), 2e-9);
      close(static_cast<PsiFormerComplexWF*>(components[i])->observeCoulombEnergy(*particles[i], 0).kinetic,
            oracle_kinetic[i], 2e-8);
    }
    CHECK(metrics.snapshot().allocation_calls == collective_before.allocation_calls);
    CHECK(metrics.snapshot().free_calls == collective_before.free_calls);
    refresh();
    initialize_collective_outputs();
    gpu.mw_evaluateMultiParticleMove(team, targets, moves, log_ratios, collective_g, collective_l);
    resource.rewind(0);
    CHECK_NOTHROW(gpu.releaseResource(resource, team)); // unresolved valid loan discards component markers only
    for (std::size_t i = 0; i < 3; ++i)
      CHECK_NOTHROW(static_cast<PsiFormerComplexWF*>(components[i])->observeCoulombEnergy(*particles[i], 0));
    resource.rewind(0);
    gpu.acquireResource(resource, team);
    CHECK_THROWS(gpu.mw_accept_rejectMultiParticleMove(team, targets, moves, reject_all));
    refresh();
    const auto publication = gpu.gpuParameterPublicationRequirements();
    auto candidate = gpu.snapshotParameters();
    candidate.values[plan.parameter(ps::ParameterRole::BACKFLOW_UP_WEIGHT, ps::NO_ATTENTION_BLOCK,
        ps::HeadTensorComponent::IMAG).begin] += .01;
    controls.set("PSIFORMER_GPU_COMPLEX_TILE_CAPACITY", "7");
    CHECK(gpu.publishBudgetedGpuParameters(candidate, candidate.version, publication.device_bytes, publication.host_bytes) == candidate.version + 1);
    CHECK(cpu.publishParameters(candidate, candidate.version) == candidate.version + 1);
    CHECK_THROWS(gpu.observeCoulombEnergy(p0, 0));
    initialize_collective_outputs();
    CHECK_THROWS(gpu.mw_evaluateMultiParticleMove(team, targets, moves, log_ratios, collective_g, collective_l));
    for (const auto value : log_ratios) CHECK(value == (C{9, 7}));
    CHECK(gpu.gpuParameterPublicationRequirements().device_bytes == publication.device_bytes);
    CHECK_THROWS(gpu.publishBudgetedGpuParameters(candidate, candidate.version, publication.device_bytes, publication.host_bytes));
    refresh();
    controls.set("PSIFORMER_GPU_COMPLEX_TILE_CAPACITY", "2");
  }
  CHECK(metrics.snapshot().live_bytes == outside.live_bytes);
  CHECK(metrics.snapshot().failed_frees == outside.failed_frees);
}
#endif
#ifdef ENABLE_CUDA
TEST_CASE("PsiFormer paired inference crowds own independent leased lanes",
          "[wavefunction][psiformer][complex_gpu_crowds]")
{
  namespace dev = ps::device;
  int devices = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices)
  { WARN("NOT_RUN: complex crowds require a HIP device"); return; }
  REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
  { WARN("NOT_RUN: complex crowds require a CUDA device"); return; }
  REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
  ScopedComplexGpuControls controls;
  controls.set("PSIFORMER_GPU_ACTIVE_INFERENCE_CROWDS", "2");
  std::size_t capacity = 2, target = 0;
  SECTION("dense FULL tile and tail") {}
  SECTION("B1 target K3 tail") { capacity = 1; target = 3; }
  controls.set("PSIFORMER_GPU_COMPLEX_TILE_CAPACITY", std::to_string(capacity));
  auto& metrics = dev::deviceMemoryMetrics(ordinal);
  const auto outside = metrics.snapshot();
  {
    const auto lattice = cell();
    const SimulationCell simulation_cell(lattice);
    std::array<ParticleSet, 6> p{electrons(simulation_cell), electrons(simulation_cell), electrons(simulation_cell),
                               electrons(simulation_cell), electrons(simulation_cell), electrons(simulation_cell)};
    auto extra_p = electrons(simulation_cell);
    auto source = ions(simulation_cell);
    const auto initialized = parameters();
    const auto walkers = target ? std::size_t{1} : std::size_t{3};
    for (std::size_t i = 0; i < 2 * walkers; ++i)
    { p[i].R[i % 4][i % 3] += .013 * i; p[i].update(); }
    const std::array<double, 3> twist{.17, -.11, .07};
    ps::ExecutionEnvironment env;
    env.boundary = ps::BoundaryCondition::PERIODIC;
    env.geometry_feature_policy = ps::GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    env.periodic_axes = {true, true, true};
    env.head_representation = ps::HeadRepresentation::PAIRED_REAL_COMPLEX;
    env.amplitude_scalar_domain = ps::ScalarDomain::COMPLEX;
    for (int a = 0; a < 3; ++a)
      for (int d = 0; d < 3; ++d) env.lattice_vectors[a][d] = lattice.R(a, d);
    const auto plan = ps::PsiFormerExecutionPlan::fromParameters(initialized, initialized.model_shape, env);
    const auto spatial = target ? dev::estimateComplexSpatialTargetCost(plan, target)
                                : dev::estimateComplexSpatialCost(plan, capacity);
    const auto owner_bytes = dev::estimateComplexValueCost(plan, capacity).device_bytes + spatial.device_bytes;
    if (target)
      controls.target(target, spatial.device_bytes,
          dev::estimateHeadFreeTargetPrefixCost(plan, target).charged_scalar_work_units);
    PsiFormerComplexWF gpu("crowds", initialized, p[0], source, clone_cap, twist, owner_bytes);
    PsiFormerComplexWF cpu("crowds", initialized, p[0], source, clone_cap, twist);
    std::vector<std::unique_ptr<WaveFunctionComponent>> gpu_clones, cpu_clones;
    std::vector<WaveFunctionComponent*> g{&gpu}, c{&cpu};
    for (std::size_t i = 1; i < 2 * walkers; ++i)
    {
      gpu_clones.push_back(gpu.makeClone(p[i])); cpu_clones.push_back(cpu.makeClone(p[i]));
      g.push_back(gpu_clones.back().get()); c.push_back(cpu_clones.back().get());
    }
    RefVectorWithLeader<WaveFunctionComponent> team0(*g[0]), team1(*g[walkers]);
    RefVectorWithLeader<ParticleSet> targets0(p[0]), targets1(p[walkers]);
    RefVector<ParticleSet::ParticleGradient> gradients0, gradients1;
    RefVector<ParticleSet::ParticleLaplacian> laplacians0, laplacians1;
    for (std::size_t i = 0; i < walkers; ++i)
    {
      team0.push_back(*g[i]); team1.push_back(*g[walkers + i]);
      targets0.push_back(p[i]); targets1.push_back(p[walkers + i]);
      gradients0.push_back(p[i].G); gradients1.push_back(p[walkers + i].G);
      laplacians0.push_back(p[i].L); laplacians1.push_back(p[walkers + i].L);
    }
    ResourceCollection prototype("paired concurrent"); gpu.createResource(prototype);
    ResourceCollection resource0(prototype), resource1(prototype), missing("missing");
    // This failure occurs after incrementing the family count; both subsequent loans must fit.
    CHECK_THROWS(gpu.acquireResource(missing, team0));
    const auto before_resources = metrics.snapshot();
    ResourceCollectionTeamLock<WaveFunctionComponent> loan0(resource0, team0), loan1(resource1, team1);
    CHECK(metrics.snapshot().live_bytes - before_resources.live_bytes == 2 * owner_bytes);
    // Two lanes * two trunks upload only their initial active-electron index arrays.
    CHECK(metrics.snapshot().uploaded_bytes - before_resources.uploaded_bytes == 4 * capacity * sizeof(std::size_t));
    auto extra = gpu.makeClone(extra_p);
    RefVectorWithLeader<WaveFunctionComponent> third(*extra, {*extra});
    ResourceCollection resource2(prototype);
    const auto denied = metrics.snapshot();
    CHECK_THROWS(extra->acquireResource(resource2, third));
    CHECK(metrics.snapshot().allocation_calls == denied.allocation_calls);
    CHECK_THROWS(gpu.gpuDerivativeRequirements(1, 1, 1, 7));
    RuntimeOptions options; TrialWaveFunction unsupported(options);
    RefVectorWithLeader<TrialWaveFunction> trials(unsupported, {unsupported});
    RefVectorWithLeader<ParticleSet> single(p[0], {p[0]});
    CHECK_THROWS(gpu.makeBudgetedGpuDerivativeOperator(trials, single, 0, 0, 1, 1, 7,
        64 * 1024 * 1024, 64 * 1024 * 1024, 1000000000));
    CHECK(metrics.snapshot().allocation_calls == denied.allocation_calls);
    const auto run = [&](std::size_t crowd, std::shared_future<void> start, int repeats) {
#ifdef QMC_CUDA2HIP
      dev::checkRuntime(hipSetDevice(ordinal), "bind paired concurrent worker");
#else
      dev::checkRuntime(cudaSetDevice(ordinal), "bind paired concurrent worker");
#endif
      start.wait();
      auto& leader = *static_cast<PsiFormerComplexWF*>(g[crowd * walkers]);
      for (int repeat = 0; repeat < repeats; ++repeat)
      {
        for (std::size_t i = crowd * walkers; i < (crowd + 1) * walkers; ++i)
        { p[i].G = 0; p[i].L = 0; }
        if (!crowd)
          leader.mw_evaluateLog(team0, targets0, gradients0, laplacians0);
        else
          leader.mw_evaluateLog(team1, targets1, gradients1, laplacians1);
      }
    };
    const auto pair = [&](int repeats) {
      std::promise<void> begin; const auto start = begin.get_future().share();
      auto first = std::async(std::launch::async, [&] { run(0, start, repeats); });
      auto second = std::async(std::launch::async, [&] { run(1, start, repeats); });
      begin.set_value(); first.get(); second.get();
    };
    const auto compare = [&] {
      for (std::size_t i = 0; i < 2 * walkers; ++i)
      {
        const auto gradient = p[i].G; const auto laplacian = p[i].L;
        close(g[i]->get_log_value(), evaluate(*c[i], p[i]));
        for (std::size_t e = 0; e < 4; ++e)
        {
          for (int d = 0; d < 3; ++d) close(gradient[e][d], p[i].G[e][d]);
          close(laplacian[e], p[i].L[e]);
        }
      }
    };
    pair(1); compare();
    const auto warm = metrics.snapshot();
    pair(3); compare();
    CHECK(metrics.snapshot().allocation_calls == warm.allocation_calls);
    CHECK(metrics.snapshot().free_calls == warm.free_calls);
    // One poisoned lane cannot alter its peer's owner or outputs; retry reuses both lanes.
    const auto saved = p[walkers - 1].R[0][0];
    p[walkers - 1].R[0][0] = std::numeric_limits<double>::quiet_NaN();
    std::promise<void> begin; const auto start = begin.get_future().share();
    auto failed = std::async(std::launch::async, [&] { run(0, start, 1); });
    auto healthy = std::async(std::launch::async, [&] { run(1, start, 1); });
    begin.set_value(); CHECK_THROWS(failed.get()); CHECK_NOTHROW(healthy.get());
    p[walkers - 1].R[0][0] = saved;
    pair(1); compare();
    auto changed = gpu.snapshotParameters();
    changed.values[0] += .002;
    const auto imaginary = std::find_if(initialized.layouts.begin(), initialized.layouts.end(),
        [](const auto& leaf) { return leaf.name == "w_imag"; });
    REQUIRE(imaginary != initialized.layouts.end());
    changed.values[imaginary->begin] += .003;
    const auto required = gpu.gpuParameterPublicationRequirements();
    CHECK(gpu.publishBudgetedGpuParameters(changed, changed.version, required.device_bytes, required.host_bytes) == 1);
    CHECK(cpu.publishParameters(changed, changed.version) == 1);
    CHECK_THROWS(gpu.publishBudgetedGpuParameters(changed, changed.version, required.device_bytes, required.host_bytes));
    const auto before_rebind = metrics.snapshot();
    pair(2); compare(); // Both idle resource lanes adopt the exact changed-weight image.
    CHECK(metrics.snapshot().allocation_calls == before_rebind.allocation_calls);
    std::printf("COMPLEX_CROWDS PASS capacity=%zu target=%zu W=%zu extra_native_bytes=%zu warm_allocations=0\n",
        capacity, target, walkers, 2 * owner_bytes);
  }
  CHECK(metrics.snapshot().live_bytes == outside.live_bytes);
  CHECK(metrics.snapshot().failed_frees == outside.failed_frees);
}
#endif
TEST_CASE("PsiFormer paired accepted FULL reuse preserves additive products and invalidation",
          "[wavefunction][psiformer][complex_full_reuse]")
{
  const SimulationCell simulation_cell(cell());
  auto source = ions(simulation_cell);
  const auto initialized = parameters();
  auto& manager = getGlobalTimerManager();
  struct RestoreThreshold
  {
    TimerManager<NewTimer>& manager;
    std::string threshold;
    ~RestoreThreshold() { manager.set_timer_threshold(threshold); }
  } restore{manager, manager.get_timer_threshold_string()};
  manager.set_timer_threshold(timer_level_fine);
  auto numerical_calls = [&] {
    TimerManager<NewTimer>::FlatProfileData profile;
    manager.collate_flat_profile(nullptr, profile);
    long calls = 0;
    for (const auto* name : {"PsiFormer::CPU_spatial_complete", "PsiFormer::GPU_HOST_evaluate_complete"})
      if (const auto entry = profile.nameList.find(name); entry != profile.nameList.end())
        calls += profile.callList[entry->second];
    return calls;
  };
  std::vector<std::pair<std::size_t, std::size_t>> backends{{0, 1}};
#ifdef ENABLE_CUDA
  int devices = 0;
#ifdef QMC_CUDA2HIP
  const bool available = hipGetDeviceCount(&devices) == hipSuccess && devices;
#else
  const bool available = cudaGetDeviceCount(&devices) == cudaSuccess && devices;
#endif
  if (available)
  {
    backends.emplace_back(64 * 1024 * 1024, 1);
    if (ps::device::configuredDeviceMemoryAdmission().limit_bytes)
      backends.emplace_back(64 * 1024 * 1024, 2);
    else
      WARN("NOT_RUN: paired dense FULL reuse requires explicit native aggregate device admission");
  }
  else WARN("NOT_RUN: paired FULL reuse GPU parity requires an available device");
  ScopedComplexGpuControls controls;
#endif
  for (const auto [device_cap, tile_capacity] : backends)
    for (const auto twist : {std::array<double, 3>{}, std::array<double, 3>{.17, -.11, .07}})
    {
      CAPTURE(device_cap, tile_capacity, twist);
#ifdef ENABLE_CUDA
      controls.set("PSIFORMER_GPU_COMPLEX_TILE_CAPACITY", std::to_string(tile_capacity));
#endif
      auto p0 = electrons(simulation_cell), p1 = electrons(simulation_cell);
      p1.R[0][2] += .039; p1.update();
      PsiFormerComplexWF family("reuse", initialized, p0, source, clone_cap, twist,
          device_cap, {}, "all_electron", device_cap ? 0 : clone_cap);
      auto other = family.makeClone(p1);
      auto& clone = dynamic_cast<PsiFormerComplexWF&>(*other);
      evaluate(family, p0); evaluate(clone, p1);
      std::array<ParticleSet::ParticleGradient, 2> reference_g{p0.G, p1.G};
      std::array<ParticleSet::ParticleLaplacian, 2> reference_l{p0.L, p1.L};
      std::array<C, 2> reference_logs{family.get_log_value(), clone.get_log_value()};
      RefVectorWithLeader<WaveFunctionComponent> team(family, {family, clone});
      RefVectorWithLeader<ParticleSet> targets(p0, {p0, p1});
      ParticleSet::ParticleGradient g0(4), g1(4);
      ParticleSet::ParticleLaplacian l0(4), l1(4);
      RefVector<ParticleSet::ParticleGradient> gs{g0, g1};
      RefVector<ParticleSet::ParticleLaplacian> ls{l0, l1};
      const C goffset{.3, -.2}, loffset{-.4, .1};
      auto cached = [&] {
        g0 = g1 = goffset; l0 = l1 = loffset;
        const auto before = numerical_calls();
        family.mw_evaluateGL(team, targets, gs, ls, false);
        CHECK(numerical_calls() == before);
        for (int walker = 0; walker < 2; ++walker)
        {
          close(team[walker].get_log_value(), reference_logs[walker], 2e-12);
          for (int e = 0; e < 4; ++e)
          {
            close(ls[walker].get()[e], reference_l[walker][e] + loffset, 2e-12);
            for (int d = 0; d < 3; ++d)
              close(gs[walker].get()[e][d], reference_g[walker][e][d] + goffset, 2e-12);
          }
        }
      };
      ResourceCollection resources("accepted FULL reuse");
      family.createResource(resources);
      ResourceCollectionTeamLock<WaveFunctionComponent> lease(resources, team);
      cached();
      g0 = goffset; l0 = loffset;
      const auto before_scalar = numerical_calls();
      close(family.evaluateGL(p0, g0, l0, false), reference_logs[0], 2e-12);
      CHECK(numerical_calls() == before_scalar);
      for (int e = 0; e < 4; ++e)
      {
        close(l0[e], reference_l[0][e] + loffset, 2e-12);
        for (int d = 0; d < 3; ++d)
          close(g0[e][d], reference_g[0][e][d] + goffset, 2e-12);
      }
      // VALUE observations reuse numerical scratch but cannot revoke accepted FULL products.
      p0.makeVirtualMoves(p0.R[0] + ParticleSet::PosType{.01, .02, 0});
      std::vector<C> virtual_ratios(4);
      family.evaluateRatiosAlltoOne(p0, virtual_ratios);
      cached();
      // Failure in the last walker cannot partially add the first walker's products.
      g0 = g1 = goffset; l0 = l1 = loffset;
      l1[3] = C{std::numeric_limits<double>::infinity(), 0};
      CHECK_THROWS(family.mw_evaluateGL(team, targets, gs, ls, false));
      CHECK(g0[0][0] == goffset); CHECK(l0[0] == loffset); CHECK(g1[0][0] == goffset);
      g0 = g1 = goffset; l0 = l1 = loffset;
      auto alias = gs; alias[1] = std::ref(g0);
      CHECK_THROWS(family.mw_evaluateGL(team, targets, alias, ls, false));
      cached();
      // A selected accepted FULL proposal immediately supplies the next observation.
      const MCMultiParticleMoves<CoordsType::POS> moves({0, 2, 4}, {1, 2, 1, 2},
          {p0.R[1] + ParticleSet::PosType{.003, 0, 0}, p0.R[2] + ParticleSet::PosType{0, -.004, 0},
           p1.R[1] + ParticleSet::PosType{.005, 0, 0}, p1.R[2] + ParticleSet::PosType{0, -.006, 0}});
      g0 = g1 = C{}; l0 = l1 = C{};
      std::vector<C> log_ratios(2);
      family.mw_evaluateMultiParticleMove(team, targets, moves, log_ratios, gs, ls);
      CHECK_THROWS(family.mw_evaluateGL(team, targets, gs, ls, false));
      reference_g[0] = g0; reference_l[0] = l0; reference_logs[0] += log_ratios[0];
      p0.R[1] = moves.slice(0).proposedPosition(0); p0.R[2] = moves.slice(0).proposedPosition(1); p0.update();
      family.mw_accept_rejectMultiParticleMove(team, targets, moves, std::vector<bool>{true, false});
      cached();
      // Force fresh independently of valid accepted products.
      g0 = g1 = C{}; l0 = l1 = C{};
      const auto before_fresh = numerical_calls();
      family.mw_evaluateGL(team, targets, gs, ls, true);
      CHECK(numerical_calls() > before_fresh);
      cached();
      // Configuration mismatch must recompute, including all output rows.
      p1.R[3][0] += .023; p1.update();
      auto oracle_p = electrons(simulation_cell); oracle_p.R = p1.R; oracle_p.update();
      PsiFormerComplexWF oracle("fresh_reuse", initialized, oracle_p, source, clone_cap, twist);
      reference_logs[1] = evaluate(oracle, oracle_p); reference_g[1] = oracle_p.G; reference_l[1] = oracle_p.L;
      g0 = g1 = C{}; l0 = l1 = C{};
      const auto before_changed = numerical_calls();
      family.mw_evaluateGL(team, targets, gs, ls, false);
      CHECK(numerical_calls() > before_changed);
      close(clone.get_log_value(), reference_logs[1], 2e-12);
      for (int e = 0; e < 4; ++e)
      {
        close(l1[e], reference_l[1][e], 2e-12);
        for (int d = 0; d < 3; ++d)
          close(g1[e][d], reference_g[1][e][d], 2e-12);
      }
      reference_logs[1] = clone.get_log_value(); reference_g[1] = g1; reference_l[1] = l1;
      cached();
      // A scalar VALUE acceptance has no FULL cache and must force a new spatial evaluation.
      p0.makeMove(0, ParticleSet::PosType{.007, 0, 0}); family.ratio(p0, 0);
      family.acceptMove(p0, 0); p0.acceptMove(0); p0.donePbyP();
      g0 = C{}; l0 = C{};
      const auto before_value = numerical_calls();
      family.evaluateGL(p0, g0, l0, false);
      CHECK(numerical_calls() > before_value);
      reference_logs[0] = family.get_log_value(); reference_g[0] = g0; reference_l[0] = l0;
      cached();
      if (!device_cap)
      {
        const auto snapshot = family.snapshotParameters(); family.publishParameters(snapshot, snapshot.version);
        g0 = g1 = C{}; l0 = l1 = C{};
        const auto before_version = numerical_calls();
        family.mw_evaluateGL(team, targets, gs, ls, false);
        CHECK(numerical_calls() > before_version);
        reference_logs = {family.get_log_value(), clone.get_log_value()};
        reference_g = {g0, g1}; reference_l = {l0, l1};
        cached();
      }
      const auto source_x = source.R[0][0]; source.R[0][0] += .01;
      g0 = g1 = goffset; l0 = l1 = loffset;
      CHECK_THROWS(family.mw_evaluateGL(team, targets, gs, ls, false));
      CHECK(g0[0][0] == goffset); CHECK(l0[0] == loffset);
      source.R[0][0] = source_x;
      cached();
    }
}

TEST_CASE("PsiFormer paired all-to-one VALUE observes complete replacements without state publication",
          "[wavefunction][psiformer][complex_all_to_one]")
{
  const auto lattice = cell();
  const SimulationCell simulation_cell(lattice);
  auto source = ions(simulation_cell);
  const auto initialized = parameters();
  std::vector<std::size_t> device_caps{0};
#ifdef ENABLE_CUDA
  int devices = 0;
#ifdef QMC_CUDA2HIP
  const bool available = hipGetDeviceCount(&devices) == hipSuccess && devices;
#else
  const bool available = cudaGetDeviceCount(&devices) == cudaSuccess && devices;
#endif
  if (available) device_caps.push_back(64 * 1024 * 1024);
  else WARN("NOT_RUN: paired all-to-one GPU parity requires an available device");
  ScopedComplexGpuControls controls;
#endif
  for (const auto device_cap : device_caps)
    for (const std::array<double, 3> twist : {std::array<double, 3>{0, 0, 0}, {.17, -.11, .07}})
    {
      CAPTURE(device_cap, twist);
      auto p = electrons(simulation_cell);
      auto foreign = electrons(simulation_cell);
      PsiFormerComplexWF component("all_to_one", initialized, p, source, clone_cap, twist, device_cap);
      CHECK(component.gpuInferenceEnabled() == (device_cap != 0));
      const auto accepted_log = evaluate(component, p);
      const auto accepted_positions = p.R;
      const auto accepted_g = p.G;
      const auto accepted_l = p.L;
      const auto snapshot = component.snapshotParameters();
      const auto storage = component.numericStorage();
      const auto accepted_energy = component.observeCoulombEnergy(p, 0);
      auto point = p.R[0]; point[0] += .037; point[2] -= .019;
      std::vector<C> ratios(p.getTotalNum(), C{71, -9});
      p.makeVirtualMoves(point);
#ifdef ENABLE_CUDA
      ps::device::DeviceMemoryMetrics* transfer_metrics = nullptr;
      ps::device::DeviceMemorySnapshot before_stream;
      if (device_cap)
      {
        int ordinal = -1;
#ifdef QMC_CUDA2HIP
        REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
        REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
        transfer_metrics = &ps::device::deviceMemoryMetrics(ordinal);
        before_stream = transfer_metrics->snapshot();
      }
#endif
      component.evaluateRatiosAlltoOne(p, ratios);
#ifdef ENABLE_CUDA
      if (transfer_metrics)
      {
        const auto after_stream = transfer_metrics->snapshot();
        const std::size_t n = p.getTotalNum();
        CHECK(after_stream.uploaded_bytes - before_stream.uploaded_bytes == 3 * sizeof(double) * (3 * n - 1));
        CHECK(after_stream.allocation_calls == before_stream.allocation_calls);
      }
#endif
      for (int electron = 0; electron < p.getTotalNum(); ++electron)
      {
        auto explicit_p = electrons(simulation_cell);
        explicit_p.R[electron] = point;
        explicit_p.update();
        // Independent complete FULL evaluation at replaced coordinates, with a fresh CPU owner.
        PsiFormerComplexWF oracle("explicit", initialized, explicit_p, source, clone_cap, twist);
        close(ratios[electron], std::exp(evaluate(oracle, explicit_p) - accepted_log), 2e-12);
      }
      const auto expected = ratios;
      component.evaluateRatiosAlltoOne(p, ratios);
      CHECK(ratios == expected);
      close(component.observeCoulombEnergy(p, 0).kinetic, accepted_energy.kinetic, 2e-12);
      auto translated = p.R[0];
      for (int d = 0; d < 3; ++d) translated[d] += lattice.R(0, d);
      p.makeVirtualMoves(translated);
      component.evaluateRatiosAlltoOne(p, ratios);
      close(ratios[0], std::exp(C{0, 2 * std::acos(-1.) * twist[0]}), 2e-12);
      close(ratios[1], C{}, 2e-12); // translated coincident same-spin rows
      const auto saved = ratios;
      auto bad = point; bad[1] = std::numeric_limits<double>::quiet_NaN();
      p.makeVirtualMoves(bad);
      CHECK_THROWS(component.evaluateRatiosAlltoOne(p, ratios)); CHECK(ratios == saved);
      p.makeVirtualMoves(point);
      std::vector<C> short_output(1, C{71, -9});
      CHECK_THROWS(component.evaluateRatiosAlltoOne(p, short_output));
      CHECK(short_output == std::vector<C>(1, C{71, -9}));
      CHECK_THROWS(component.evaluateRatiosAlltoOne(foreign, ratios)); CHECK(ratios == saved);
      const auto source_x = source.R[0][0]; source.R[0][0] += .01;
      CHECK_THROWS(component.evaluateRatiosAlltoOne(p, ratios)); CHECK(ratios == saved);
      source.R[0][0] = source_x;
      // Failed observation preserves the pending proposal so its original acceptance still succeeds.
      p.makeMove(2, ParticleSet::PosType{.01, 0, 0});
      const auto proposal_ratio = component.ratio(p, 2);
      CHECK_THROWS(component.evaluateRatiosAlltoOne(p, ratios)); CHECK(ratios == saved);
      CHECK(component.get_log_value() == accepted_log);
      for (int electron = 0; electron < p.getTotalNum(); ++electron)
        for (int d = 0; d < 3; ++d)
          CHECK(p.R[electron][d] == accepted_positions[electron][d]);
      CHECK(p.G == accepted_g); CHECK(p.L == accepted_l);
      CHECK(component.snapshotParameters().values == snapshot.values);
      CHECK(component.numericStorage().fingerprint == storage.fingerprint);
      CHECK(component.numericStorage().allocated_bytes == storage.allocated_bytes);
      component.acceptMove(p, 2); p.acceptMove(2); p.donePbyP();
      close(std::exp(component.get_log_value() - accepted_log), proposal_ratio, 2e-12);
      evaluate(component, p);
      p.makeVirtualMoves(point);
      component.evaluateRatiosAlltoOne(p, ratios);
      if (!device_cap)
      {
        const auto current = component.snapshotParameters();
        component.publishParameters(current, current.version);
        const auto before_stale = ratios;
        CHECK_THROWS(component.evaluateRatiosAlltoOne(p, ratios)); CHECK(ratios == before_stale);
        evaluate(component, p);
        component.evaluateRatiosAlltoOne(p, ratios);
      }
    }
  auto p = electrons(simulation_cell);
  auto zero = initialized;
  for (const auto& leaf : zero.layouts)
    if (leaf.module.find("Backflow") != std::string::npos)
      std::fill(zero.values.begin() + leaf.begin, zero.values.begin() + leaf.end, 0.);
  PsiFormerComplexWF nodal("zero_all_to_one", zero, p, source, clone_cap);
  CHECK_THROWS(evaluate(nodal, p));
  p.makeVirtualMoves(p.R[0]);
  std::vector<C> sentinel(4, C{71, -9});
  CHECK_THROWS(nodal.evaluateRatiosAlltoOne(p, sentinel));
  CHECK(sentinel == std::vector<C>(4, C{71, -9}));
}

TEST_CASE("PsiFormer paired virtual VALUE preserves accepted state and the sole Bloch phase",
          "[wavefunction][psiformer][complex_virtual_value]")
{
  const auto lattice = cell();
  const SimulationCell simulation_cell(lattice);
  auto p = electrons(simulation_cell), sibling_p = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  const auto initialized = parameters();
  const std::array<double, 3> twist{.17, -.11, .07};
  PsiFormerComplexWF component("virtual", initialized, p, source, clone_cap, twist);
  const auto accepted_log = evaluate(component, p);
  const auto accepted_g = p.G;
  const auto accepted_l = p.L;
  const auto snapshot = component.snapshotParameters();
  const auto storage = component.numericStorage();
  auto sibling_base = component.makeClone(sibling_p);
  auto& sibling = dynamic_cast<PsiFormerComplexWF&>(*sibling_base);
  close(evaluate(sibling, sibling_p), accepted_log);
  // Both spin rows use absolute positions, including an exact lattice translation.
  for (int electron : {0, 2})
  {
    VirtualParticleSet vp(p);
    std::vector<ParticleSet::PosType> positions{p.R[electron], p.R[electron], p.R[electron]};
    positions[1][0] += .031;
    positions[1][2] -= .017;
    for (int d = 0; d < 3; ++d)
      positions[2][d] += lattice.R(0, d);
    vp.makeMovesAbsolute(p, electron, {positions.data(), positions.size()});
    std::vector<C> ratios(3, C{71, -9});
    component.evaluateRatios(vp, ratios);
    close(ratios[0], C{1, 0});
    close(ratios[2], std::exp(C{0, 2 * std::acos(-1.) * twist[0]}));
    for (std::size_t q = 0; q < positions.size(); ++q)
    {
      auto fresh_p = electrons(simulation_cell);
      fresh_p.R[electron] = positions[q];
      fresh_p.update();
      PsiFormerComplexWF fresh("independent", initialized, fresh_p, source, clone_cap, twist);
      close(ratios[q], std::exp(evaluate(fresh, fresh_p) - accepted_log));
    }
    const auto expected = ratios;
    component.evaluateRatios(vp, ratios);
    CHECK(ratios == expected);
    vp.R[2][0] = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS(component.evaluateRatios(vp, ratios));
    CHECK(ratios == expected);
    vp.makeMovesAbsolute(p, electron, {positions.data(), positions.size()});
    vp.refPtcl = 4;
    CHECK_THROWS(component.evaluateRatios(vp, ratios));
    CHECK(ratios == expected);
    vp.refPtcl = electron;
    std::vector<C> short_output(2, C{71, -9});
    CHECK_THROWS(component.evaluateRatios(vp, short_output));
    CHECK(short_output == std::vector<C>(2, C{71, -9}));
    VirtualParticleSet foreign(sibling_p);
    foreign.makeMovesAbsolute(sibling_p, electron, {positions.data(), positions.size()});
    CHECK_THROWS(component.evaluateRatios(foreign, ratios));
    p.makeMove(electron, ParticleSet::PosType{.01, 0, 0});
    component.ratio(p, electron);
    CHECK_THROWS(component.evaluateRatios(vp, ratios));
    component.restore(electron);
    p.rejectMove(electron);
    component.evaluateRatios(vp, ratios); // rejected calls do not poison the accepted denominator
    CHECK(ratios == expected);
    vp.R[0] = p.R[electron + 1]; // equal same-spin rows: theoretical node, not an IEEE-zero claim
    component.evaluateRatios(vp, ratios);
    close(ratios[0], C{});
    vp.R[0] = positions[0];
    component.evaluateRatios(vp, ratios);
    CHECK(ratios == expected);
  }
  CHECK(component.get_log_value() == accepted_log);
  CHECK(p.G == accepted_g);
  CHECK(p.L == accepted_l);
  CHECK(p.getActivePtcl() == -1);
  CHECK(component.snapshotParameters().values == snapshot.values);
  CHECK(component.numericStorage().fingerprint == storage.fingerprint);
  CHECK(component.numericStorage().allocated_bytes == storage.allocated_bytes);
  close(evaluate(sibling, sibling_p), accepted_log);
  VirtualParticleSet scratch0(p), scratch1(sibling_p);
  RefVectorWithLeader<WaveFunctionComponent> capture_components(component, {component, sibling});
  RefVectorWithLeader<ParticleSet> capture_particles(p, {p, sibling_p});
  RefVectorWithLeader<VirtualParticleSet> capture_scratch(scratch0, {scratch0, scratch1});
  const std::vector<std::size_t> offsets{0, 3, 4};
  const std::vector<VirtualParticleBatch::Segment> segments{{0, 0}, {1, 2}};
  std::vector<ParticleSet::PosType> points{p.R[0], p.R[0], p.R[0], sibling_p.R[2]};
  points[1][0] += .023; points[3][2] -= .019;
  const VirtualParticleBatch batch(2, offsets, segments, points);
  std::vector<C> captured(4, C{71, -9});
  const auto capture = [&] {
    return component.mw_evaluateVirtualRatios(capture_components, capture_particles, capture_scratch, batch, captured);
  };
  const auto stamp = capture();
  REQUIRE(stamp.isVersioned());
  const auto expected_capture = captured;
  CHECK(capture() == stamp);
  CHECK(captured == expected_capture);
  close(captured[0], C{1, 0}); close(captured[2], C{1, 0});
  for (std::size_t segment = 0; segment < 2; ++segment)
  {
    const auto slice = batch.slice(segment);
    auto& scratch = capture_scratch[segment];
    scratch.makeMovesAbsolute(capture_particles[segment], slice.electronId(), slice.positions());
    std::vector<C> scalar(slice.size());
    capture_components[segment].evaluateRatios(scratch, scalar);
    for (std::size_t q = 0; q < slice.size(); ++q) close(captured[slice.flatOffset() + q], scalar[q]);
  }
  RefVectorWithLeader<VirtualParticleSet> aliases(scratch0, {scratch0, scratch0});
  CHECK_THROWS(component.mw_evaluateVirtualRatios(capture_components, capture_particles, aliases, batch, captured));
  CHECK(captured == expected_capture);
  PsiFormerComplexWF foreign_capture("foreign", initialized, sibling_p, source, clone_cap, twist);
  evaluate(foreign_capture, sibling_p);
  RefVectorWithLeader<WaveFunctionComponent> foreign_components(component, {component, foreign_capture});
  CHECK_THROWS(component.mw_evaluateVirtualRatios(foreign_components, capture_particles, capture_scratch, batch, captured));
  CHECK(captured == expected_capture);
  p.makeMove(0, ParticleSet::PosType{.01, 0, 0}); component.ratio(p, 0);
  CHECK_THROWS(capture()); CHECK(captured == expected_capture);
  component.restore(0); p.rejectMove(0);
  points.back()[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(capture()); CHECK(captured == expected_capture);
  points.back()[0] = sibling_p.R[2][0];
  CHECK(capture() == stamp);
  VirtualParticleSet vp(p);
  vp.makeMoves(p, 0, {ParticleSet::PosType{.02, 0, 0}});
  std::vector<C> sentinel(1, C{71, -9});
  component.publishParameters(snapshot, snapshot.version);
  CHECK_THROWS(component.evaluateRatios(vp, sentinel));
  CHECK(sentinel == std::vector<C>(1, C{71, -9}));
  evaluate(component, p);
  component.evaluateRatios(vp, sentinel);
  evaluate(sibling, sibling_p);
  CHECK(capture() != stamp);
  // A zero BASE is undefined; no determinant inverse or undefined score is needed to reject it.
  auto zero = initialized;
  for (const auto& leaf : zero.layouts)
    if (leaf.module.find("Backflow") != std::string::npos)
      std::fill(zero.values.begin() + leaf.begin, zero.values.begin() + leaf.end, 0.);
  PsiFormerComplexWF nodal("zero", zero, p, source, clone_cap, twist);
  CHECK_THROWS(evaluate(nodal, p)); // zero BASE cannot publish the FULL accepted reference
  const auto saved = sentinel;
  CHECK_THROWS(nodal.evaluateRatios(vp, sentinel));
  CHECK(sentinel == saved);
#ifdef ENABLE_CUDA
  int devices = 0;
#ifdef QMC_CUDA2HIP
  const bool available = hipGetDeviceCount(&devices) == hipSuccess && devices;
#else
  const bool available = cudaGetDeviceCount(&devices) == cudaSuccess && devices;
#endif
  if (available)
  {
    ScopedComplexGpuControls controls;
    auto gpu_p = electrons(simulation_cell);
    PsiFormerComplexWF gpu("virtual_gpu", initialized, gpu_p, source, clone_cap, twist, 64 * 1024 * 1024);
    REQUIRE(gpu.gpuInferenceEnabled());
    const auto gpu_log = evaluate(gpu, gpu_p);
    VirtualParticleSet gpu_vp(gpu_p);
    std::vector<ParticleSet::PosType> positions{p.R[0], p.R[0], p.R[1]};
    positions[0][0] += .02;
    for (int d = 0; d < 3; ++d)
      positions[1][d] += lattice.R(0, d);
    gpu_vp.makeMovesAbsolute(gpu_p, 0, {positions.data(), positions.size()});
    vp.makeMovesAbsolute(p, 0, {positions.data(), positions.size()});
    std::vector<C> gpu_ratios(3), cpu_ratios(3);
    component.evaluateRatios(vp, cpu_ratios);
    int ordinal = -1;
#ifdef QMC_CUDA2HIP
    REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
    REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
    auto& metrics = ps::device::deviceMemoryMetrics(ordinal);
    const auto warm = metrics.snapshot();
    gpu.evaluateRatios(gpu_vp, gpu_ratios);
    CHECK(metrics.snapshot().uploaded_bytes - warm.uploaded_bytes ==
        3 * sizeof(double) * (gpu_p.getTotalNum() + positions.size()));
    for (std::size_t q = 0; q < gpu_ratios.size(); ++q)
      close(gpu_ratios[q], cpu_ratios[q]);
    CHECK(gpu.get_log_value() == gpu_log);
    CHECK(metrics.snapshot().allocation_calls == warm.allocation_calls);
    const std::vector<std::size_t> gpu_offsets{0, positions.size()};
    const std::vector<VirtualParticleBatch::Segment> gpu_segments{{0, 0}};
    const VirtualParticleBatch gpu_batch(1, gpu_offsets, gpu_segments, positions);
    RefVectorWithLeader<WaveFunctionComponent> gpu_components(gpu, {gpu});
    RefVectorWithLeader<ParticleSet> gpu_particles(gpu_p, {gpu_p});
    RefVectorWithLeader<VirtualParticleSet> gpu_scratch(gpu_vp, {gpu_vp});
    const auto gpu_stamp = gpu.mw_evaluateVirtualRatios(gpu_components, gpu_particles, gpu_scratch, gpu_batch, gpu_ratios);
    CHECK(gpu_stamp.isVersioned());
    for (std::size_t q = 0; q < gpu_ratios.size(); ++q) close(gpu_ratios[q], cpu_ratios[q]);
    CHECK(metrics.snapshot().allocation_calls == warm.allocation_calls);
    std::printf("COMPLEX_VIRTUAL_CAPTURE GPU PASS versioned=1 Q=3 warm_allocations=0\n");
    std::printf("COMPLEX_VIRTUAL_VALUE GPU PASS Q=3 warm_allocations=0\n");
  }
  else
    WARN("NOT_RUN: paired virtual VALUE GPU parity requires an available device");
#endif
}
#ifdef ENABLE_CUDA
TEST_CASE("PsiFormer paired virtual envelopes remain request local across ragged walkers",
          "[wavefunction][psiformer][complex_virtual_envelope]")
{
  int devices = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices) { WARN("NOT_RUN: envelope reuse requires GPU"); return; }
  REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) { WARN("NOT_RUN: envelope reuse requires GPU"); return; }
  REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
  if (!ps::device::configuredDeviceMemoryAdmission().limit_bytes)
  { WARN("NOT_RUN: envelope reuse requires explicit aggregate device admission"); return; }
  ScopedComplexGpuControls controls;
  controls.set("PSIFORMER_GPU_COMPLEX_VIRTUAL_TILE_CAPACITY", "3");
  const SimulationCell simulation_cell(cell());
  auto source = ions(simulation_cell);
  const auto initialized = parameters(true, false, 7, 64);
  constexpr std::size_t host_cap = std::size_t(512) * 1024 * 1024;
  constexpr std::size_t device_cap = std::size_t(2) * 1024 * 1024 * 1024;
  auto p = electrons(simulation_cell, 32, 32), p1 = electrons(simulation_cell, 32, 32);
  for (int e = 0; e < 64; ++e)
    for (int d = 0; d < 3; ++d)
    {
      p.R[e][d] = 1.4 + 1.1 * std::sin(.29 * (3 * e + d + 1));
      p1.R[e][d] = p.R[e][d] + .013 * (d + 1);
    }
  p.update(); p1.update();
  const std::array<double, 3> twist{.17, -.11, .07};
  PsiFormerComplexWF ordinary("ordinary", initialized, p, source, host_cap, twist, device_cap);
  auto ordinary_clone = ordinary.makeClone(p1);
  auto& ordinary1 = dynamic_cast<PsiFormerComplexWF&>(*ordinary_clone);
  controls.set("PSIFORMER_GPU_COMPLEX_VIRTUAL_ENVELOPE_REUSE", "1");
  PsiFormerComplexWF cached("cached", initialized, p, source, host_cap, twist, device_cap);
  auto cached_clone = cached.makeClone(p1);
  auto& cached1 = dynamic_cast<PsiFormerComplexWF&>(*cached_clone);
  const auto old_log = evaluate(ordinary, p), old_log1 = evaluate(ordinary1, p1);
  CHECK(evaluate(cached, p) == old_log); CHECK(evaluate(cached1, p1) == old_log1);
  const auto accepted_g = p.G;
  const auto accepted_l = p.L;
  RefVectorWithLeader<ParticleSet> particles(p, {p, p1});
  RefVectorWithLeader<WaveFunctionComponent> old_components(ordinary, {ordinary, ordinary1});
  RefVectorWithLeader<WaveFunctionComponent> new_components(cached, {cached, cached1});
  VirtualParticleSet v0(p), v1(p1);
  v0.makeMovesAbsolute(p, 0, {&p.R[0], 1});
  v1.makeMovesAbsolute(p1, 63, {&p1.R[63], 1});
  const auto scratch0 = v0.R, scratch1 = v1.R;
  RefVectorWithLeader<VirtualParticleSet> scratch(v0, {v0, v1});
  // Walker1 returns noncontiguously, and row changes cross a capacity3/tail boundary.
  const std::vector<std::size_t> offsets{0, 2, 6, 7, 13};
  const std::vector<VirtualParticleBatch::Segment> segments{{1, 63}, {0, 0}, {0, 32}, {1, 1}};
  std::vector<ParticleSet::PosType> points;
  for (std::size_t s = 0; s < segments.size(); ++s)
    for (std::size_t q = offsets[s]; q < offsets[s + 1]; ++q)
    {
      points.push_back(particles[segments[s].walkerId()].R[segments[s].electronId()]);
      points.back()[q % 3] += .0017 * (q + 1);
    }
  const VirtualParticleBatch batch(2, offsets, segments, points);
  std::vector<C> expected(points.size()), actual(points.size(), C{71, -9});
  const auto call = [&] { return cached.mw_evaluateVirtualRatios(new_components, particles, scratch, batch, actual); };
  ordinary.mw_evaluateVirtualRatios(old_components, particles, scratch, batch, expected);
  auto& metrics = ps::device::deviceMemoryMetrics(ordinal);
  const auto warm = metrics.snapshot();
  const auto stamp = call();
  REQUIRE(stamp.isVersioned()); CHECK(actual == expected);
  CHECK(call() == stamp); CHECK(actual == expected);
  CHECK(metrics.snapshot().allocation_calls == warm.allocation_calls);
  CHECK(metrics.snapshot().free_calls == warm.free_calls);
  CHECK(cached.get_log_value() == old_log); CHECK(cached1.get_log_value() == old_log1);
  CHECK(p.G == accepted_g); CHECK(p.L == accepted_l);
  REQUIRE(v0.R.size() == scratch0.size()); REQUIRE(v1.R.size() == scratch1.size());
  for (std::size_t q = 0; q < scratch0.size(); ++q)
    for (int d = 0; d < 3; ++d) CHECK(v0.R[q][d] == scratch0[q][d]);
  for (std::size_t q = 0; q < scratch1.size(); ++q)
    for (int d = 0; d < 3; ++d) CHECK(v1.R[q][d] == scratch1[q][d]);
  const auto saved = actual;
  const auto last = points.back();
  points.back()[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(call()); CHECK(actual == saved); points.back() = last;
  CHECK(call() == stamp); CHECK(actual == expected);
  p.makeMove(0, ParticleSet::PosType{.01, 0, 0}); cached.ratio(p, 0);
  CHECK_THROWS(call()); CHECK(actual == saved);
  cached.restore(0); p.rejectMove(0);
  CHECK(call() == stamp); CHECK(actual == expected);
  auto changed = cached.snapshotParameters(); changed.values[0] += .002;
  const auto publication = cached.gpuParameterPublicationRequirements();
  REQUIRE(cached.publishBudgetedGpuParameters(changed, changed.version,
      publication.device_bytes, publication.host_bytes) == 1);
  CHECK_THROWS(call()); CHECK(actual == saved);
  evaluate(cached, p); evaluate(cached1, p1);
  const auto rebound = metrics.snapshot();
  CHECK(call() != stamp);
  CHECK(metrics.snapshot().allocation_calls == rebound.allocation_calls);
  // Source mutation is rejected atomically, rather than authorizing a stale base bank.
  const auto before_source = actual;
  source.R[0][0] += .01; source.update();
  CHECK_THROWS(call()); CHECK(actual == before_source);
  std::printf("COMPLEX_VIRTUAL_ENVELOPE_CHECKS_COMPLETED Ne=64 T=3 Q=13 W=2 old_GPU_bits=checked CSR_order=checked "
              "scratch_unchanged=checked warm_allocations=checked publication_rebind=checked atomic_failure=checked\n");
}

TEST_CASE("PsiFormer paired virtual tiles pack independent ragged walker replacements",
          "[wavefunction][psiformer][complex_virtual_tile]")
{
  int devices = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
  if (hipGetDeviceCount(&devices) != hipSuccess || !devices) { WARN("NOT_RUN: virtual tile requires GPU"); return; }
  REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) { WARN("NOT_RUN: virtual tile requires GPU"); return; }
  REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
  if (!ps::device::configuredDeviceMemoryAdmission().limit_bytes)
  { WARN("NOT_RUN: virtual tile requires explicit aggregate device admission"); return; }
  ScopedComplexGpuControls controls;
  auto& metrics = ps::device::deviceMemoryMetrics(ordinal);
  const auto outside = metrics.snapshot();
  const SimulationCell simulation_cell(cell());
  auto source = ions(simulation_cell);
  const auto initialized = parameters();
  for (const auto twist : {std::array<double, 3>{}, std::array<double, 3>{.17, -.11, .07}})
    for (std::size_t capacity : {std::size_t(1), std::size_t(2), std::size_t(4)})
    {
      CAPTURE(capacity, twist);
      controls.set("PSIFORMER_GPU_COMPLEX_VIRTUAL_TILE_CAPACITY", std::to_string(capacity));
      auto p = electrons(simulation_cell), p1 = electrons(simulation_cell);
      p1.R[1][2] += .037; p1.update();
      PsiFormerComplexWF gpu("tile", initialized, p, source, clone_cap, twist, 64 * 1024 * 1024);
      auto clone = gpu.makeClone(p1);
      auto& gpu1 = dynamic_cast<PsiFormerComplexWF&>(*clone);
      const auto log0 = evaluate(gpu, p), log1 = evaluate(gpu1, p1);
      const auto accepted_g = p.G, accepted_g1 = p1.G;
      const auto accepted_l = p.L, accepted_l1 = p1.L;
      const auto storage = gpu.numericStorage();
      RefVectorWithLeader<WaveFunctionComponent> components(gpu, {gpu, gpu1});
      RefVectorWithLeader<ParticleSet> particles(p, {p, p1});
      VirtualParticleSet v0(p), v1(p1);
      v0.makeMovesAbsolute(p, 0, {&p.R[0], 1});
      v1.makeMovesAbsolute(p1, 2, {&p1.R[2], 1});
      const auto scratch0 = v0.R, scratch1 = v1.R;
      RefVectorWithLeader<VirtualParticleSet> scratch(v0, {v0, v1});
      const std::vector<std::size_t> offsets{0, 2, 5, 9};
      const std::vector<VirtualParticleBatch::Segment> segments{{1, 2}, {0, 0, true, 0}, {1, 3}};
      std::vector<ParticleSet::PosType> points;
      std::vector<C> expected;
      for (std::size_t segment = 0; segment < segments.size(); ++segment)
        for (std::size_t q = offsets[segment]; q < offsets[segment + 1]; ++q)
        {
          const auto walker = segments[segment].walkerId(), electron = segments[segment].electronId();
          auto absolute = particles[walker].R[electron];
          absolute[0] += .009 * (q + 1); absolute[2] -= .003 * q;
          if (q == 8)
            for (int d = 0; d < 3; ++d) absolute[d] += cell().R(0, d);
          points.push_back(absolute);
          auto fresh_p = electrons(simulation_cell);
          for (int e = 0; e < 4; ++e) fresh_p.R[e] = particles[walker].R[e];
          fresh_p.R[electron] = absolute; fresh_p.update();
          PsiFormerComplexWF fresh("independent", initialized, fresh_p, source, clone_cap, twist);
          expected.push_back(std::exp(evaluate(fresh, fresh_p) - (walker ? log1 : log0)));
        }
      const VirtualParticleBatch batch(2, offsets, segments, points);
      std::vector<C> output(points.size(), C{71, -9});
      const auto call = [&] { return gpu.mw_evaluateVirtualRatios(components, particles, scratch, batch, output); };
      const auto warm = metrics.snapshot();
      const auto stamp = call();
      REQUIRE(stamp.isVersioned());
      for (std::size_t q = 0; q < output.size(); ++q) close(output[q], expected[q]);
      CHECK(call() == stamp);
      CHECK(metrics.snapshot().allocation_calls == warm.allocation_calls);
      const std::vector<std::size_t> empty_offsets{0};
      const std::vector<VirtualParticleBatch::Segment> empty_segments;
      const std::vector<ParticleSet::PosType> empty_points;
      const VirtualParticleBatch empty(2, empty_offsets, empty_segments, empty_points);
      std::vector<C> empty_output;
      CHECK(gpu.mw_evaluateVirtualRatios(components, particles, scratch, empty, empty_output) == stamp);
      CHECK(empty_output.empty());
      const std::vector<std::size_t> bad_offsets{0, 2, 2, 5, 9};
      const std::vector<VirtualParticleBatch::Segment> bad_segments{{1, 2}, {0, 1}, {0, 0}, {1, 3}};
      CHECK_THROWS(VirtualParticleBatch(2, bad_offsets, bad_segments, points));
      REQUIRE(v0.R.size() == scratch0.size()); REQUIRE(v1.R.size() == scratch1.size());
      for (std::size_t q = 0; q < scratch0.size(); ++q)
        for (int d = 0; d < 3; ++d) CHECK(v0.R[q][d] == scratch0[q][d]);
      for (std::size_t q = 0; q < scratch1.size(); ++q)
        for (int d = 0; d < 3; ++d) CHECK(v1.R[q][d] == scratch1[q][d]);
      CHECK(gpu.get_log_value() == log0); CHECK(gpu1.get_log_value() == log1);
      CHECK(p.G == accepted_g); CHECK(p1.G == accepted_g1);
      CHECK(p.L == accepted_l); CHECK(p1.L == accepted_l1);
      CHECK(gpu.numericStorage().fingerprint == storage.fingerprint);
      const auto saved = output;
      const auto last = points.back(); points.back()[0] = std::numeric_limits<double>::quiet_NaN();
      CHECK_THROWS(call()); CHECK(output == saved); points.back() = last;
      RefVectorWithLeader<VirtualParticleSet> aliases(v0, {v0, v0});
      CHECK_THROWS(gpu.mw_evaluateVirtualRatios(components, particles, aliases, batch, output));
      CHECK(output == saved);
      p.makeMove(0, ParticleSet::PosType{.01, 0, 0}); gpu.ratio(p, 0);
      CHECK_THROWS(call()); CHECK(output == saved);
      gpu.restore(0); p.rejectMove(0);
      CHECK(call() == stamp);
      auto changed = gpu.snapshotParameters(); changed.values[0] += .002;
      const auto budget = gpu.gpuParameterPublicationRequirements();
      REQUIRE(gpu.publishBudgetedGpuParameters(changed, changed.version, budget.device_bytes, budget.host_bytes) == 1);
      CHECK_THROWS(call()); CHECK(output == saved);
      evaluate(gpu, p); evaluate(gpu1, p1);
      const auto after_publication = metrics.snapshot();
      CHECK(call() != stamp);
      CHECK(metrics.snapshot().allocation_calls == after_publication.allocation_calls);
      // The stronger ON-only numeric bound rejects before image or optional allocation.
      const auto before_cap = metrics.snapshot();
      const auto combined = storage.shared_gpu_bytes + storage.shared_gpu_host_bytes;
      CHECK_THROWS(PsiFormerComplexWF("capped", initialized, p, source, clone_cap, twist, combined - 1));
      CHECK(metrics.snapshot().allocation_calls == before_cap.allocation_calls);
      std::printf("COMPLEX_VIRTUAL_TILE PASS T=%zu Q=9 W=2 independent_CPU=1 scratch_unchanged=1 warm_allocations=0\n", capacity);
    }
  // A competing native reservation can win after local numeric preflight.  Admit
  // the candidate's old image/base owners, but leave no optional-tile headroom;
  // destruction must undo every partial candidate owner without harming a peer.
  {
    controls.set("PSIFORMER_GPU_COMPLEX_VIRTUAL_TILE_CAPACITY", "0");
    auto p = electrons(simulation_cell);
    PsiFormerComplexWF peer("peer", initialized, p, source, clone_cap, {}, 64 * 1024 * 1024);
    const auto peer_log = evaluate(peer, p);
    const auto inventory = peer.numericStorage();
    const auto baseline = metrics.snapshot();
    const auto policy = ps::device::configuredDeviceMemoryAdmission();
    const auto available = *policy.limit_bytes - policy.external_reserve_bytes - baseline.live_bytes;
    const auto old_owners = inventory.shared_gpu_bytes + inventory.shared_image_bytes;
    REQUIRE(available > old_owners);
    {
      auto competing = metrics.reserveAllocation(available - old_owners);
      controls.set("PSIFORMER_GPU_COMPLEX_VIRTUAL_TILE_CAPACITY", "4");
      CHECK_THROWS(PsiFormerComplexWF("partial", initialized, p, source, clone_cap, {}, 64 * 1024 * 1024));
      CHECK(metrics.snapshot().live_bytes == baseline.live_bytes);
      CHECK(metrics.snapshot().allocation_calls > baseline.allocation_calls);
    }
    CHECK(peer.get_log_value() == peer_log);
    CHECK_NOTHROW(evaluate(peer, p));
  }
  CHECK(metrics.snapshot().live_bytes == outside.live_bytes);
  CHECK(metrics.snapshot().failed_frees == outside.failed_frees);
}

TEST_CASE("PsiFormer paired captured-grid ECP binds existing numerical ownership",
          "[wavefunction][psiformer][complex_captured_ecp]")
{
  using namespace wftrain;
  int count = 0, ordinal = -1;
#ifdef QMC_CUDA2HIP
  if (hipGetDeviceCount(&count) != hipSuccess || !count) { WARN("NOT_RUN: captured ECP requires GPU"); return; }
  REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
  if (cudaGetDeviceCount(&count) != cudaSuccess || !count) { WARN("NOT_RUN: captured ECP requires GPU"); return; }
  REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
  ScopedComplexGpuControls controls;
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell, 3, 1), p1 = electrons(simulation_cell, 3, 1);
  p1.R[1][2] += .037; p1.update();
  auto source = ions(simulation_cell);
  auto initialized = parameters();
  initialized.model_shape.spin_up_electrons = 3; initialized.model_shape.spin_down_electrons = 1;
  const std::array<double, 3> twist{.17, -.11, .07};
  RuntimeOptions options;
  TrialWaveFunction trial(options);
  auto owner = std::make_unique<PsiFormerComplexWF>("captured", initialized, p, source, clone_cap, twist, 64 * 1024 * 1024);
  auto& gpu = *owner; trial.addComponent(std::move(owner));
  auto trial1 = trial.makeClone(p1);
  auto& gpu1 = dynamic_cast<PsiFormerComplexWF&>(*trial1->getOrbitals()[0]);
  evaluate(gpu, p); evaluate(gpu1, p1);
  const auto accepted = gpu.snapshotParameters();
  const auto accepted_log = gpu.get_log_value();
  RefVectorWithLeader<TrialWaveFunction> trials(trial, {trial, *trial1});
  RefVectorWithLeader<ParticleSet> particles(p, {p, p1});
  RefVectorWithLeader<WaveFunctionComponent> components(gpu, {gpu, gpu1});
  VirtualParticleSet v0(p), v1(p1);
  RefVectorWithLeader<VirtualParticleSet> scratch(v0, {v0, v1});
  const std::vector<std::size_t> offsets{0, 4, 7, 9};
  const std::vector<VirtualParticleBatch::Segment> segments{{1, 2}, {0, 0}, {1, 3}};
  std::vector<ParticleSet::PosType> points;
  for (std::size_t s = 0; s < segments.size(); ++s)
    for (std::size_t q = offsets[s]; q < offsets[s + 1]; ++q)
    {
      points.push_back(particles[segments[s].walkerId()].R[segments[s].electronId()]);
      points.back()[0] += .009 * (q + 1); points.back()[2] -= .003 * q;
    }
  const VirtualParticleBatch batch(2, offsets, segments, points);
  std::vector<C> ratios(9);
  const auto stamp = gpu.mw_evaluateVirtualRatios(components, particles, scratch, batch, ratios);
  REQUIRE(stamp.isVersioned());
  const auto requirement = gpu.gpuDerivativeRequirements(2, 13, 3, 7);
  const auto make = [&](std::size_t extra = 65536) {
    return gpu.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, 13, 3, 7,
        requirement.device_bytes, requirement.host_bytes + extra, requirement.matrix_work);
  };
  auto op = make();
  const auto replayFor = [&](const StreamingDerivativeOperator& target) {
    auto replay = std::make_unique<ComplexCapturedReplay>(target);
    replay->positions.resize(24);
    for (std::size_t b = 0; b < 2; ++b)
      for (std::size_t e = 0; e < 4; ++e)
        for (int d = 0; d < 3; ++d) replay->positions[12 * b + 3 * e + d] = particles[b].R[e][d];
    for (std::size_t s = 0; s < segments.size(); ++s)
    {
      replay->segments_.push_back({offsets[s], offsets[s + 1] - offsets[s],
          std::size_t(segments[s].walkerId()), std::size_t(segments[s].electronId()), false, std::size_t(-1)});
      for (std::size_t q = offsets[s]; q < offsets[s + 1]; ++q)
        replay->grid.push_back({std::size_t(segments[s].walkerId()), std::size_t(segments[s].electronId()),
            {points[q][0], points[q][1], points[q][2]}, C{(q % 2 ? -.13 : .17) * (q + 1), 0}, ratios[q]});
    }
    replay->tiles_.push_back({0, 0, 9, 0, 3, 0, 1, batch.fingerprint()}); replay->stamps.push_back(stamp);
    return replay;
  };
  const auto before_bind = op->storageDiagnostics();
  auto& metrics = ps::device::deviceMemoryMetrics(ordinal);
  const auto device_before_bind = metrics.snapshot();
  auto bad = replayFor(*op); ++bad->version;
  CHECK_THROWS(op->bindNonLocalECPJVPReplay(std::move(bad)));
  bad = replayFor(*op); bad->stamps[0] = {};
  CHECK_THROWS(op->bindNonLocalECPJVPReplay(std::move(bad)));
  bad = replayFor(*op); bad->grid[0].bare_weight.imag(.01);
  CHECK_THROWS(op->bindNonLocalECPJVPReplay(std::move(bad)));
  bad = replayFor(*op); bad->stamps.push_back(stamp); bad->tiles_[0].stamp_offset = 1;
  CHECK_THROWS(op->bindNonLocalECPJVPReplay(std::move(bad)));
  bad = replayFor(*op); bad->stamps.push_back(stamp);
  CHECK_THROWS(op->bindNonLocalECPJVPReplay(std::move(bad)));
  bad = replayFor(*op); bad->grid.resize(12, bad->grid.front());
  CHECK_THROWS(op->bindNonLocalECPJVPReplay(std::move(bad)));
  bad = replayFor(*op); bad->grid[0].bare_weight = {std::numeric_limits<double>::infinity(), 0};
  CHECK_THROWS(op->bindNonLocalECPJVPReplay(std::move(bad)));
  CHECK(op->storageDiagnostics().storage_fingerprint == before_bind.storage_fingerprint);
  auto denied = make(0);
  CHECK_THROWS(denied->bindNonLocalECPJVPReplay(replayFor(*denied)));
  CHECK(denied->storageDiagnostics().retained_numeric_bytes == before_bind.retained_numeric_bytes);
  denied.reset();
  auto replay = replayFor(*op);
  const auto grid = replay->grid;
  const auto replay_bytes = replay->storageDiagnostics().retained_numeric_bytes;
  op->bindNonLocalECPJVPReplay(std::move(replay));
  CHECK(metrics.snapshot().allocation_calls == device_before_bind.allocation_calls);
  CHECK(metrics.snapshot().live_bytes == device_before_bind.live_bytes);
  CHECK_THROWS(op->bindNonLocalECPJVPReplay(replayFor(*op)));
  REQUIRE(op->capabilities().maximum_vjp_channels == 3);
  REQUIRE(op->capabilities().local_energy_term_mask ==
      (localEnergyTermBit(LocalEnergyTerm::KINETIC) | localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)));
  const auto storage = op->storageDiagnostics();
  CHECK(storage.shared_workspace_bytes == before_bind.shared_workspace_bytes);
  // Binding adds only references, weights and base coefficients, not full virtual coordinates.
  CHECK(storage.retained_numeric_bytes == before_bind.retained_numeric_bytes + replay_bytes +
      grid.size() * sizeof(std::size_t) + (op->sampleCount() + grid.size()) * sizeof(C));
  CHECK(storage.allocation_generation == 2);
  // Independent CPU score contraction uses actual complete physical ratios on the identical grid.
  std::vector<std::unique_ptr<ParticleSet>> all;
  for (std::size_t b = 0; b < 11; ++b)
  {
    const auto reference = b < 2 ? b : grid[b - 2].sample;
    all.push_back(std::make_unique<ParticleSet>(particles[reference]));
    if (b >= 2)
      for (int d = 0; d < 3; ++d) all.back()->R[grid[b - 2].electron][d] = grid[b - 2].replacement_position[d];
    all.back()->update();
  }
  PsiFormerComplexWF cpu("captured", initialized, *all[0], source, clone_cap, twist);
  std::vector<std::unique_ptr<WaveFunctionComponent>> copies;
  RefVectorWithLeader<WaveFunctionComponent> cpu_components(cpu, {cpu});
  RefVectorWithLeader<ParticleSet> cpu_particles(*all[0], {*all[0]});
  std::vector<C> logs{evaluate(cpu, *all[0])};
  for (std::size_t b = 1; b < all.size(); ++b)
  {
    copies.push_back(cpu.makeClone(*all[b])); logs.push_back(evaluate(*copies.back(), *all[b]));
    cpu_components.push_back(*copies.back()); cpu_particles.push_back(*all[b]);
  }
  auto oracle = cpu.makeBudgetedStreamingDerivativeOperator(cpu_components, cpu_particles, 7, 11, 7,
      cpu.streamingDerivativeRequiredBytes(11));
  const auto& schema = gpu.parameterSchema();
  std::vector<C> coefficients{{.7, -.2}, {-.4, .3}}, folded(11), direction(schema.parameterCount());
  for (std::size_t i = 0; i < direction.size(); ++i) direction[i] = .003 * std::sin(double(i + 1));
  auto used = make();
  ComplexSampleSink used_jvp;
  used->applyScoreJVP({schema, 0, {direction.data(), direction.size()}}, used_jvp);
  CHECK_THROWS(used->bindNonLocalECPJVPReplay(replayFor(*used)));
  used.reset();
  VJPCoefficientChannel channel{"ecp", DerivativeProduct::LOCAL_ENERGY_VJP,
      {schema.providerId(), schema.fingerprint(), 0, 7, 11, {coefficients.data(), 2}}, localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)};
  const auto& cs = cpu.parameterSchema();
  VJPCoefficientChannel score{"score", DerivativeProduct::SCORE_VJP,
      {cs.providerId(), cs.fingerprint(), 0, 7, 11, {folded.data(), folded.size()}}};
  ComplexStreamingSink expected, actual;
  for (const auto adjoint : {DerivativeAdjoint::TRANSPOSE, DerivativeAdjoint::HERMITIAN})
  {
    std::fill(folded.begin(), folded.end(), C{});
    for (std::size_t q = 0; q < grid.size(); ++q)
    {
      const auto weight = grid[q].bare_weight * std::exp(logs[q + 2] - logs[grid[q].sample]);
      close(weight, grid[q].bare_weight * grid[q].complete_ratio, 3e-9);
      folded[q + 2] = coefficients[grid[q].sample] * (adjoint == DerivativeAdjoint::HERMITIAN ? std::conj(weight) : weight);
      folded[grid[q].sample] -= folded[q + 2];
    }
    expected.reset(); actual.reset();
    oracle->applyVJPs({&score, 1}, adjoint, expected);
    const auto warm = metrics.snapshot(); op->applyVJPs({&channel, 1}, adjoint, actual);
    CHECK(metrics.snapshot().allocation_calls == warm.allocation_calls);
    for (std::size_t i = 0; i < direction.size(); ++i) close(actual.values[0][i], expected.values[0][i], 3e-9);
  }
  ComplexSampleSink jvp;
  op->applyLocalEnergyJVP({schema, 0, {direction.data(), direction.size()}}, localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP), jvp);
  C dual{}, dot{};
  for (std::size_t b = 0; b < 2; ++b) dual += std::conj(coefficients[b]) * jvp.values[b];
  for (std::size_t i = 0; i < direction.size(); ++i) dot += std::conj(actual.values[0][i]) * direction[i];
  close(dual, dot, 3e-9);
  RefVectorWithLeader<WaveFunctionComponent> base_components(cpu, {cpu, *copies[0]});
  RefVectorWithLeader<ParticleSet> base_particles(*all[0], {*all[0], *all[1]});
  auto base_oracle = cpu.makeBudgetedStreamingDerivativeOperator(base_components, base_particles, 7, 11, 7,
      cpu.streamingDerivativeRequiredBytes(2));
  auto kinetic_channel = channel;
  kinetic_channel.local_energy_term_mask = localEnergyTermBit(LocalEnergyTerm::KINETIC);
  for (const auto mask : {localEnergyTermBit(LocalEnergyTerm::KINETIC),
      localEnergyTermBit(LocalEnergyTerm::KINETIC) | localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)})
    for (const auto adjoint : {DerivativeAdjoint::TRANSPOSE, DerivativeAdjoint::HERMITIAN})
    {
      ComplexStreamingSink kinetic_expected, ecp_expected, complete_actual;
      base_oracle->applyVJPs({&kinetic_channel, 1}, adjoint, kinetic_expected);
      op->applyVJPs({&channel, 1}, adjoint, ecp_expected);
      auto complete_channel = channel; complete_channel.local_energy_term_mask = mask;
      const auto warm = metrics.snapshot();
      op->applyVJPs({&complete_channel, 1}, adjoint, complete_actual);
      CHECK(metrics.snapshot().allocation_calls == warm.allocation_calls);
      for (std::size_t i = 0; i < direction.size(); ++i)
        close(complete_actual.values[0][i], kinetic_expected.values[0][i] +
            (mask == kinetic_channel.local_energy_term_mask ? C{} : ecp_expected.values[0][i]), 3e-9);
      ComplexSampleSink complete_jvp;
      op->applyLocalEnergyJVP({schema, 0, {direction.data(), direction.size()}}, mask, complete_jvp);
      C lhs{}, rhs{};
      for (std::size_t b = 0; b < 2; ++b)
        lhs += (adjoint == DerivativeAdjoint::HERMITIAN ? std::conj(coefficients[b]) : coefficients[b]) * complete_jvp.values[b];
      for (std::size_t i = 0; i < direction.size(); ++i)
        rhs += (adjoint == DerivativeAdjoint::HERMITIAN ? std::conj(complete_actual.values[0][i]) :
            complete_actual.values[0][i]) * direction[i];
      close(lhs, rhs, 3e-9);
    }
  // EO/O ordering is caller-defined; one complete channel shares the same base score and final receipt.
  std::vector<C> eo{{.23, -.19}, {-.31, .07}}, overlap{{-.12, .29}, {.43, -.17}};
  for (std::size_t energy_slot = 0; energy_slot < 3; ++energy_slot)
    for (const auto adjoint : {DerivativeAdjoint::TRANSPOSE, DerivativeAdjoint::HERMITIAN})
    {
      std::array<VJPCoefficientChannel, 3> request{channel, channel, channel};
      for (std::size_t c = 0; c < 3; ++c)
      {
        request[c].id = c == 0 ? "first" : c == 1 ? "second" : "third";
        if (c == energy_slot) request[c].local_energy_term_mask = localEnergyTermBit(LocalEnergyTerm::KINETIC);
        else
        {
          request[c].product = DerivativeProduct::SCORE_VJP;
          request[c].local_energy_term_mask = 0;
          const auto& values = c == (energy_slot + 1) % 3 ? eo : overlap;
          request[c].coefficients.values = {values.data(), values.size()};
        }
      }
      ComplexStreamingSink cpu_channels, ecp_channel, gpu_channels;
      base_oracle->applyVJPs({request.data(), request.size()}, adjoint, cpu_channels);
      op->applyVJPs({&channel, 1}, adjoint, ecp_channel);
      request[energy_slot].local_energy_term_mask |= localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);
      const auto warm = metrics.snapshot();
      op->applyVJPs({request.data(), request.size()}, adjoint, gpu_channels);
      CHECK(metrics.snapshot().allocation_calls == warm.allocation_calls);
      for (std::size_t c = 0; c < 3; ++c)
        for (std::size_t i = 0; i < direction.size(); ++i)
          close(gpu_channels.values[c][i], cpu_channels.values[c][i] +
              (c == energy_slot ? ecp_channel.values[0][i] : C{}), 3e-9);
      request[(energy_slot + 1) % 3] = request[energy_slot];
      request[(energy_slot + 1) % 3].id = "duplicate_energy";
      gpu_channels.reset();
      CHECK_THROWS(op->applyVJPs({request.data(), request.size()}, adjoint, gpu_channels));
      CHECK(gpu_channels.state() == DerivativeSinkState::POISONED);
      for (const auto& values : gpu_channels.values)
        CHECK(std::all_of(values.begin(), values.end(), [](C value) { return value == C{}; }));
    }
  std::printf("COMPLEX_CAPTURED_COMPLETE_C3 PASS B=2 Q=9 energy_slots=3 EO_O=1 both_adjoints=1 one_owner=1\n");
  actual.reset(); actual.fail_next = true;
  CHECK_THROWS(op->applyVJPs({&channel, 1}, DerivativeAdjoint::HERMITIAN, actual));
  actual.reset(); CHECK_NOTHROW(op->applyVJPs({&channel, 1}, DerivativeAdjoint::HERMITIAN, actual));
  CHECK(op->storageDiagnostics().storage_fingerprint == storage.storage_fingerprint);
  channel.local_energy_term_mask = 0x80000000u;
  actual.reset(); CHECK_THROWS(op->applyVJPs({&channel, 1}, DerivativeAdjoint::TRANSPOSE, actual));
  channel.local_energy_term_mask = localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);
  p.makeMove(0, ParticleSet::PosType{.01, 0, 0}); gpu.ratio(p, 0);
  actual.reset(); CHECK_THROWS(op->applyVJPs({&channel, 1}, DerivativeAdjoint::TRANSPOSE, actual));
  gpu.restore(0); p.rejectMove(0);
  CHECK(gpu.snapshotParameters().values == accepted.values); CHECK(gpu.get_log_value() == accepted_log);
  // Publication during COMPLETE response chunks must not replace the leased old generation.
  ComplexStreamingSink kinetic_expected;
  base_oracle->applyVJPs({&kinetic_channel, 1}, DerivativeAdjoint::HERMITIAN, kinetic_expected);
  channel.local_energy_term_mask |= localEnergyTermBit(LocalEnergyTerm::KINETIC);
  bool published = false;
  actual.reset(); actual.callback = [&] {
    if (published) return;
    published = true;
    const auto budget = gpu.gpuParameterPublicationRequirements();
    CHECK(gpu.publishBudgetedGpuParameters(accepted, 0, budget.device_bytes, budget.host_bytes) == 1);
  };
  op->applyVJPs({&channel, 1}, DerivativeAdjoint::HERMITIAN, actual);
  CHECK(published);
  for (std::size_t i = 0; i < direction.size(); ++i)
    close(actual.values[0][i], expected.values[0][i] + kinetic_expected.values[0][i], 3e-9);
  actual.reset(); CHECK_THROWS(op->applyVJPs({&channel, 1}, DerivativeAdjoint::TRANSPOSE, actual));
  std::printf("COMPLEX_CAPTURED_ECP PASS B=2 Q=9 C3_slot0 C1_complete CPU_allP duality stamp lease retry stale\n");
  // A new current-generation selective lane must actually omit the kinetic owner.
  op.reset();
  evaluate(gpu, p); evaluate(gpu1, p1);
  const auto current = gpu.snapshotParameters();
  const auto current_stamp = gpu.mw_evaluateVirtualRatios(components, particles, scratch, batch, ratios);
  constexpr auto selective_scope = PsiFormerComplexWF::GpuDerivativeScope::SCORE_ECP_ONLY;
  // Score covariance needs accepted B only, not capacity for any virtual grid.
  const auto accepted_requirement = gpu.gpuDerivativeRequirements(2, 2, 3, 7, selective_scope);
  {
    auto accepted_only = gpu.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, 2, 3, 7,
        accepted_requirement.device_bytes, accepted_requirement.host_bytes, 1, selective_scope);
    CHECK(accepted_requirement.matrix_work == 0);
    CHECK_FALSE(accepted_only->capabilities().supports(DerivativeProduct::LOCAL_ENERGY_VJP));
    CHECK(accepted_only->capabilities().local_energy_term_mask == 0);
    auto accepted_channel = channel;
    accepted_channel.product = DerivativeProduct::SCORE_VJP;
    accepted_channel.local_energy_term_mask = 0;
    accepted_channel.coefficients.parameter_version = current.version;
    auto cpu_channel = accepted_channel;
    cpu_channel.coefficients.parameter_version = 0;
    ComplexStreamingSink accepted_result, cpu_result;
    for (const auto adjoint : {DerivativeAdjoint::TRANSPOSE, DerivativeAdjoint::HERMITIAN})
    {
      cpu_result.reset(); accepted_result.reset();
      base_oracle->applyVJPs({&cpu_channel, 1}, adjoint, cpu_result);
      accepted_only->applyVJPs({&accepted_channel, 1}, adjoint, accepted_result);
      for (std::size_t i = 0; i < direction.size(); ++i)
        close(accepted_result.values[0][i], cpu_result.values[0][i], 2e-12);
    }
    accepted_channel.product = DerivativeProduct::LOCAL_ENERGY_VJP;
    ComplexStreamingSink refused;
    CHECK_THROWS(accepted_only->applyVJPs({&accepted_channel, 1}, DerivativeAdjoint::TRANSPOSE, refused));
  }
  const auto selective_requirement = gpu.gpuDerivativeRequirements(2, 13, 3, 7, selective_scope);
  // Reverse products remain device B1; only HOST capacity grows with maxB.
  CHECK(accepted_requirement.device_bytes == selective_requirement.device_bytes);
  CHECK(accepted_requirement.host_bytes < selective_requirement.host_bytes);
  CHECK(selective_requirement.matrix_work == 0);
  CHECK(selective_requirement.device_bytes < requirement.device_bytes);
  CHECK(selective_requirement.host_bytes < requirement.host_bytes);
  CHECK_THROWS(gpu.gpuDerivativeRequirements(2, 13, 3, 7,
      static_cast<PsiFormerComplexWF::GpuDerivativeScope>(9)));
  const auto selective_make = [&](std::size_t device_cap, std::size_t host_cap) {
    return gpu.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, 13, 3, 7,
        device_cap, host_cap, 1, selective_scope);
  };
  const auto selective_cold = metrics.snapshot();
  CHECK_THROWS(selective_make(selective_requirement.device_bytes - 1, selective_requirement.host_bytes));
  CHECK_THROWS(selective_make(selective_requirement.device_bytes, selective_requirement.host_bytes - 1));
  CHECK(metrics.snapshot().allocation_calls == selective_cold.allocation_calls);
  auto selective = selective_make(selective_requirement.device_bytes, selective_requirement.host_bytes + 65536);
  CHECK(metrics.snapshot().live_bytes - selective_cold.live_bytes == selective_requirement.device_bytes);
  const auto selective_storage = selective->storageDiagnostics();
  CHECK(selective_storage.real_parameter_vectors == 3);
  CHECK(selective_storage.complex_parameter_vectors == 4);
  CHECK(selective_storage.parameter_scratch_bytes == 88 * schema.parameterCount() + 32 * 7);
  CHECK(selective_storage.retained_numeric_bytes == selective_requirement.device_bytes + selective_requirement.host_bytes);
  CHECK(selective->capabilities().supports(DerivativeProduct::SCORE_VJP));
  CHECK(selective->capabilities().supports(DerivativeProduct::SCORE_JVP));
  CHECK_FALSE(selective->capabilities().supports(DerivativeProduct::LOCAL_ENERGY_VJP));
  CHECK_FALSE(selective->capabilities().supports(DerivativeProduct::LOCAL_ENERGY_JVP));
  CHECK(selective->capabilities().local_energy_term_mask == 0);
  const auto before_mismatch = metrics.snapshot();
  CHECK_THROWS(make()); // Same family, different live preparation scope: no implicit larger owner.
  CHECK(metrics.snapshot().allocation_calls == before_mismatch.allocation_calls);
  auto bound_selective = selective_make(selective_requirement.device_bytes, selective_requirement.host_bytes + 65536);
  CHECK(bound_selective->storageDiagnostics().shared_workspace_identity == selective_storage.shared_workspace_identity);
  auto selective_channel = channel;
  selective_channel.product = DerivativeProduct::SCORE_VJP; selective_channel.local_energy_term_mask = 0;
  selective_channel.coefficients.parameter_version = current.version;
  auto cpu_score_channel = selective_channel; cpu_score_channel.coefficients.parameter_version = 0;
  ComplexStreamingSink selective_expected, selective_actual;
  for (const auto adjoint : {DerivativeAdjoint::TRANSPOSE, DerivativeAdjoint::HERMITIAN})
  {
    selective_expected.reset(); selective_actual.reset();
    base_oracle->applyVJPs({&cpu_score_channel, 1}, adjoint, selective_expected);
    selective->applyVJPs({&selective_channel, 1}, adjoint, selective_actual);
    for (std::size_t i = 0; i < direction.size(); ++i)
      close(selective_actual.values[0][i], selective_expected.values[0][i], 3e-9);
  }
  ComplexSampleSink selective_score_jvp, cpu_score_jvp;
  base_oracle->applyScoreJVP({cs, 0, {direction.data(), direction.size()}}, cpu_score_jvp);
  selective->applyScoreJVP({schema, current.version, {direction.data(), direction.size()}}, selective_score_jvp);
  for (std::size_t b = 0; b < 2; ++b) close(selective_score_jvp.values[b], cpu_score_jvp.values[b], 3e-9);
  auto selective_replay = replayFor(*bound_selective); selective_replay->stamps[0] = current_stamp;
  const auto selective_warm = metrics.snapshot();
  bound_selective->bindNonLocalECPJVPReplay(std::move(selective_replay));
  CHECK(bound_selective->capabilities().local_energy_term_mask == localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP));
  CHECK(bound_selective->capabilities().supports(DerivativeProduct::LOCAL_ENERGY_VJP));
  CHECK(bound_selective->capabilities().supports(DerivativeProduct::LOCAL_ENERGY_JVP));
  selective_channel.product = DerivativeProduct::LOCAL_ENERGY_VJP;
  selective_channel.local_energy_term_mask = localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);
  for (const auto adjoint : {DerivativeAdjoint::TRANSPOSE, DerivativeAdjoint::HERMITIAN})
  {
    std::fill(folded.begin(), folded.end(), C{});
    for (std::size_t q = 0; q < grid.size(); ++q)
    {
      const auto w = grid[q].bare_weight * grid[q].complete_ratio;
      folded[q + 2] = coefficients[grid[q].sample] * (adjoint == DerivativeAdjoint::HERMITIAN ? std::conj(w) : w);
      folded[grid[q].sample] -= folded[q + 2];
    }
    selective_expected.reset(); selective_actual.reset();
    oracle->applyVJPs({&score, 1}, adjoint, selective_expected);
    bound_selective->applyVJPs({&selective_channel, 1}, adjoint, selective_actual);
    for (std::size_t i = 0; i < direction.size(); ++i)
      close(selective_actual.values[0][i], selective_expected.values[0][i], 3e-9);
  }
  ComplexSampleSink selective_ecp_jvp;
  bound_selective->applyLocalEnergyJVP({schema, current.version, {direction.data(), direction.size()}},
      localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP), selective_ecp_jvp);
  for (std::size_t b = 0; b < 2; ++b) close(selective_ecp_jvp.values[b], jvp.values[b], 3e-9);
  const auto before_unsupported = metrics.snapshot();
  for (const auto mask : {localEnergyTermBit(LocalEnergyTerm::KINETIC),
      localEnergyTermBit(LocalEnergyTerm::KINETIC) | localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)})
  {
    selective_channel.local_energy_term_mask = mask; selective_actual.reset();
    CHECK_THROWS(bound_selective->applyVJPs({&selective_channel, 1}, DerivativeAdjoint::TRANSPOSE, selective_actual));
    ComplexSampleSink refused;
    CHECK_THROWS(bound_selective->applyLocalEnergyJVP({schema, current.version, {direction.data(), direction.size()}}, mask, refused));
  }
  CHECK(metrics.snapshot().uploaded_bytes == before_unsupported.uploaded_bytes);
  CHECK(metrics.snapshot().downloaded_bytes == before_unsupported.downloaded_bytes);
  CHECK(metrics.snapshot().allocation_calls == selective_warm.allocation_calls);
  CHECK(selective->storageDiagnostics().storage_fingerprint == selective_storage.storage_fingerprint);
  selective_channel.local_energy_term_mask = localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP);
  bool selective_published = false;
  selective_actual.reset(); selective_actual.callback = [&] {
    if (selective_published) return;
    selective_published = true;
    const auto budget = gpu.gpuParameterPublicationRequirements();
    CHECK(gpu.publishBudgetedGpuParameters(current, current.version, budget.device_bytes, budget.host_bytes) == current.version + 1);
  };
  bound_selective->applyVJPs({&selective_channel, 1}, DerivativeAdjoint::HERMITIAN, selective_actual);
  CHECK(selective_published);
  for (std::size_t i = 0; i < direction.size(); ++i)
    close(selective_actual.values[0][i], selective_expected.values[0][i], 3e-9);
  selective_actual.reset(); CHECK_THROWS(bound_selective->applyVJPs({&selective_channel, 1}, DerivativeAdjoint::TRANSPOSE, selective_actual));
  CHECK(gpu.snapshotParameters().values == accepted.values);
  std::printf("COMPLEX_SELECTIVE_RESPONSE PASS SCORE_allP=1 JVP=1 bound_ECP_bothadjoints=1 noKINETIC=1 exactbytes=1 caps_scope=1 callback_stale=1 device_saved=%zu host_saved=%zu\n",
      requirement.device_bytes - selective_requirement.device_bytes, requirement.host_bytes - selective_requirement.host_bytes);
}
TEST_CASE("PsiFormer paired captures an actual ordinary nonlocal traversal and complex energy",
          "[wavefunction][psiformer][complex_actual_ecp]")
{
  using namespace wftrain;
  int count = 0, device = -1;
#ifdef QMC_CUDA2HIP
  if (hipGetDeviceCount(&count) != hipSuccess || !count) { WARN("NOT_RUN: actual paired ECP requires GPU"); return; }
  REQUIRE(hipGetDevice(&device) == hipSuccess);
#else
  if (cudaGetDeviceCount(&count) != cudaSuccess || !count) { WARN("NOT_RUN: actual paired ECP requires GPU"); return; }
  REQUIRE(cudaGetDevice(&device) == cudaSuccess);
#endif
  ScopedComplexGpuControls controls;
  const auto virtual_capacity = GENERATE(0, 4);
  CAPTURE(virtual_capacity);
  if (virtual_capacity && !ps::device::configuredDeviceMemoryAdmission().limit_bytes)
  { WARN("NOT_RUN: native ECP virtual tile requires aggregate device admission"); return; }
  controls.set("PSIFORMER_GPU_COMPLEX_VIRTUAL_TILE_CAPACITY", std::to_string(virtual_capacity));
  const SimulationCell simulation_cell(cell());
  auto source = ions(simulation_cell), p = electrons(simulation_cell, 3, 1);
  // VP's zero-table limit means "all" when cloned; prepare the same ion table in both owners first.
  p.addTable(source); p.update();
  // An existing radial operator on the periodic numerical fixture, NOT a new native PP profile.
  auto nonlocal = std::make_unique<NonLocalECPotential>(source, p, false, true);
  auto* nl0 = nonlocal.get();
  const auto pp_path = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() /
      "tests/molecules/LiH_pp/Li.BFD.xml";
  ECPComponentBuilder builder("paired_component_ecp", OHMMS::Controller, 4, 1);
  REQUIRE(builder.read_pp_file(pp_path.string()));
  REQUIRE(builder.pp_nonloc != nullptr);
  nonlocal->addComponent(0, std::move(builder.pp_nonloc));
  testing::TestNonLocalECPotential::setTileCapacity(*nonlocal, 5);
  p.update();
  ParticleSet p1(p); p1.R[1][2] += .037; p1.update();
  auto initialized = parameters();
  initialized.model_shape.spin_up_electrons = 3; initialized.model_shape.spin_down_electrons = 1;
  const std::array<double, 3> twist{.17, -.11, .07};
  RuntimeOptions options;
  TrialWaveFunction trial(options);
  auto owner = std::make_unique<PsiFormerComplexWF>("actual_ecp", initialized, p, source, clone_cap, twist, 64 * 1024 * 1024);
  auto& gpu = *owner; trial.addComponent(std::move(owner));
  auto trial1 = trial.makeClone(p1);
  auto& gpu1 = dynamic_cast<PsiFormerComplexWF&>(*trial1->getOrbitals()[0]);
  QMCHamiltonian ham("paired_operator_observation");
  // This manually constructed radial fixture owns its exact immutable input identity, like ECPotentialBuilder.
  auto provenance = std::make_shared<QMCHamiltonian::ECPModelProvenance>();
  std::ifstream pp_input(pp_path, std::ios::binary);
  REQUIRE(pp_input.good());
  provenance->scientific_definition = "ordinary_scalar_LiBFD_rule4:" +
      std::string(std::istreambuf_iterator<char>(pp_input), std::istreambuf_iterator<char>());
  provenance->ordinary_locality_vp = true;
  ham.setECPModelProvenance(provenance);
  ham.addOperator(std::make_unique<BareKineticEnergy>(p), "Kinetic", true);
  ham.addOperator(std::move(nonlocal), "NonLocalECP", true);
  // A real, parameter-independent native term makes source-Re(W)+trial-W accounting observable.
  auto local = std::make_unique<HarmonicExternalPotential>(p);
  local->mass = local->energy = local->length = 1; local->center = 0;
  ham.addOperator(std::move(local), "LocalBaseline", true);
  const auto local_baseline = [](const ParticleSet& target) {
    double result = 0;
    for (const auto& position : target.R) result += .5 * dot(position, position);
    return result;
  };
  ham.addObservables(p);
  auto ham1 = ham.makeClone(p1, *trial1);
  auto& nl1 = dynamic_cast<NonLocalECPotential&>(*ham1->getComponent(1));
  REQUIRE(testing::TestNonLocalECPotential::sameSchema(*nl0, nl1));
  FakeRandom<QMCTraits::FullPrecRealType> rng0, rng1; rng0.set_value(.371); rng1.set_value(.371);
  nl0->setRandomGenerator(&rng0); nl1.setRandomGenerator(&rng1);
  RefVectorWithLeader<TrialWaveFunction> trials(trial, {trial, *trial1});
  RefVectorWithLeader<ParticleSet> particles(p, {p, p1});
  RefVectorWithLeader<QMCHamiltonian> hamiltonians(ham, {ham, *ham1});
  // Candidate isolation is a cold, unacquired-family operation.  Establish its
  // private owners before borrowing the source crowd's inference resources.
  ParticleSet private_p(p), private_cpu_p(p);
  auto private_owner = gpu.makeClone(private_p);
  auto& private_gpu = dynamic_cast<PsiFormerComplexWF&>(*private_owner);
  const auto isolation = private_gpu.gpuCandidateIsolationRequirements();
  private_gpu.isolateBudgetedGpuParameterState(isolation.device_bytes, isolation.host_bytes);
  ResourceCollection pr("paired actual ECP particles"), wr("paired actual ECP wavefunctions"), hr("paired actual ECP H");
  p.createResource(pr); trial.createResource(wr); ham.createResource(hr);
  ResourceCollectionTeamLock<ParticleSet> pl(pr, particles);
  ResourceCollectionTeamLock<TrialWaveFunction> wl(wr, trials);
  ResourceCollectionTeamLock<QMCHamiltonian> hl(hr, hamiltonians);
  trial.evaluateLog(p); trial1->evaluateLog(p1);
  const auto accepted = gpu.snapshotParameters();
  const auto required = gpu.gpuDerivativeRequirements(2, 128, 3, 7);
  auto op = gpu.makeBudgetedGpuDerivativeOperator(trials, particles, 19, 23, 128, 3, 7,
      required.device_bytes, required.host_bytes + 1024 * 1024, required.matrix_work);
  const auto capture_bytes = gpu.gpuNonLocalReplayRequiredBytes(*op);
  const auto shape_bytes = gpu.gpuNonLocalReplayRequirements(2, 126);
  CHECK(shape_bytes.capture_host_bytes == capture_bytes);
  // Binding owns only Q references/weights plus B coefficients; no virtual/joined coordinate bank.
  CHECK(shape_bytes.bound_host_extra_bytes == capture_bytes + 126 * sizeof(std::size_t) + 128 * sizeof(C));
  const auto removed_coordinate_bytes = 24 * p.getTotalNum() * (2 + 2 * 126);
  CHECK(removed_coordinate_bytes == 24384);
  CHECK_THROWS(gpu.gpuNonLocalReplayRequirements(0, 126));
  CHECK_THROWS(gpu.gpuNonLocalReplayRequirements(2, std::numeric_limits<std::size_t>::max()));
  auto& metrics = ps::device::deviceMemoryMetrics(device);
  const auto before = metrics.snapshot();
  CHECK_THROWS(gpu.makeBudgetedGpuNonLocalReplayConsumer(trials, particles, *op, capture_bytes - 1));
  CHECK(metrics.snapshot().allocation_calls == before.allocation_calls);
  CHECK(metrics.snapshot().uploaded_bytes == before.uploaded_bytes);
  auto consumer = gpu.makeBudgetedGpuNonLocalReplayConsumer(trials, particles, *op, capture_bytes);
  CHECK(consumer->storageDiagnostics().retained_numeric_bytes <= capture_bytes);
  CHECK(consumer->storageDiagnostics().complex_parameter_vectors == 0);
  consumer->abort(); CHECK_THROWS(consumer->releaseJVPReplay()); consumer.reset();
  consumer = gpu.makeBudgetedGpuNonLocalReplayConsumer(trials, particles, *op, capture_bytes);
  const auto& schema = gpu.parameterSchema();
  std::vector<C> coefficients{{.7, -.2}, {-.4, .3}}, direction(schema.parameterCount());
  for (std::size_t i = 0; i < direction.size(); ++i) direction[i] = .003 * std::sin(double(i + 1));
  VJPCoefficientChannel channel{"actual_nonlocal", DerivativeProduct::LOCAL_ENERGY_VJP,
      {schema.providerId(), schema.fingerprint(), 0, 19, 23, {coefficients.data(), 2}},
      localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)};
  ComplexStreamingSink replay_sink;
  const auto before_queries = metrics.snapshot();
  const auto native_requirement = ham.nativeCaptureRequirements(particles.size());
  const auto reference_requirement = gpu.fixedGridEnergyReferenceRequirements();
  CHECK(native_requirement.total_host_bytes ==
      native_requirement.new_receipt_host_bytes + native_requirement.shared_pp_host_bytes);
  CHECK_THROWS(ham.nativeCaptureRequirements(0));
  CHECK_THROWS(ham.nativeCaptureRequirements(std::numeric_limits<std::size_t>::max()));
  CHECK(reference_requirement.new_reference_host_bytes > 0);
  CHECK(reference_requirement.shared_replay_descriptor_bytes > 0);
  CHECK(metrics.snapshot().allocation_calls == before_queries.allocation_calls);
  CHECK(metrics.snapshot().uploaded_bytes == before_queries.uploaded_bytes);
  // Original Fiji identity capacity exceeded the old 1MiB literal; derive authoritative finite caps.
  std::printf("COMPLEX_FIXED_GRID_METADATA native_new=%zu PP_shared=%zu native_total=%zu ref_new=%zu replay_descriptor=%zu\n",
      native_requirement.new_receipt_host_bytes, native_requirement.shared_pp_host_bytes,
      native_requirement.total_host_bytes, reference_requirement.new_reference_host_bytes,
      reference_requirement.shared_replay_descriptor_bytes);
  CHECK_THROWS(QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(
      hamiltonians, trials, particles, *consumer, {&channel, 1}, replay_sink, native_requirement.total_host_bytes - 1));
  CHECK(replay_sink.state() == DerivativeSinkState::IDLE);
  const auto native_begin = std::chrono::steady_clock::now();
  const auto actual = QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(
      hamiltonians, trials, particles, *consumer, {&channel, 1}, replay_sink, native_requirement.total_host_bytes);
  const auto native_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - native_begin).count();
  std::printf("COMPLEX_ACTUAL_ECP_OPERATION virtual_T=%d W=2 seconds=%.9g timing=steady_clock_complete_native_capture derivative_mode=unchanged\n",
      virtual_capacity, native_seconds);
  REQUIRE(actual.native_capture);
  CHECK(actual.native_capture->hostBytes() == native_requirement.total_host_bytes);
  CHECK(actual.native_capture->scientificDefinition() == provenance->scientific_definition);
  CHECK(actual.native_capture->gridFingerprint() == actual.grid_fingerprint);
  for (std::size_t b = 0; b < 2; ++b)
    CHECK(actual.native_capture->reportedPotentials()[b] == hamiltonians[b].getLocalPotential());
  CHECK(replay_sink.state() == DerivativeSinkState::IDLE);
  auto replay = consumer->releaseJVPReplay();
  CHECK_THROWS(consumer->releaseJVPReplay());
  REQUIRE(replay->gridFingerprint() == actual.grid_fingerprint);
  const std::vector<NonLocalECPReplayPoint> grid(replay->points().begin(), replay->points().end());
  REQUIRE(grid.size() > 5); REQUIRE(grid.size() + 4 <= 128);
  CHECK(replay->tiles().size() > 1);
  // The same native grid can be contracted immediately without retaining its points.
  const auto inline_bytes = gpu.nonLocalVJPCaptureRequiredBytes(2);
  CHECK_THROWS(gpu.nonLocalVJPCaptureRequiredBytes(0));
  CHECK_THROWS(gpu.makeBudgetedNonLocalVJPConsumer(trials, particles, *op, inline_bytes - 1));
  auto inline_consumer = gpu.makeBudgetedNonLocalVJPConsumer(trials, particles, *op, inline_bytes);
  CHECK(inline_consumer->storageDiagnostics().retained_numeric_bytes <= inline_bytes);
  PreparedLocalEnergyVJPCapture inline_sink(schema, 0, channel.local_energy_term_mask);
  auto mismatched_channel = channel;
  mismatched_channel.coefficients.parameter_version = 1;
  CHECK_THROWS(QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(
      hamiltonians, trials, particles, *inline_consumer, {&mismatched_channel, 1}, inline_sink));
  CHECK(inline_sink.state() == DerivativeSinkState::IDLE);
  const auto inline_native = QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(
      hamiltonians, trials, particles, *inline_consumer, {&channel, 1}, inline_sink);
  CHECK(inline_native.grid_fingerprint == actual.grid_fingerprint);
  CHECK(inline_sink.state() == DerivativeSinkState::COMPLETE);
  CHECK_THROWS(inline_consumer->releaseJVPReplay());
  const auto inline_energies = gpu.capturedNonLocalEnergies(*inline_consumer);
  REQUIRE(inline_energies.size() == 2);
  const auto inline_response = inline_sink.release();
  CHECK(inline_response.local_energy_term_mask == channel.local_energy_term_mask);
  const auto reference_cap = reference_requirement.new_reference_host_bytes +
      reference_requirement.shared_replay_descriptor_bytes + replay->storageDiagnostics().retained_numeric_bytes +
      actual.native_capture->hostBytes(); // Shared owners ONCE, plus one new per-sample reference.
  op->bindNonLocalECPJVPReplay(std::move(replay));
  CHECK(metrics.snapshot().allocation_calls == before.allocation_calls);
  // Independent CPU amplitudes and all-P scores use precisely the accepted radial/grid data.
  std::vector<std::unique_ptr<ParticleSet>> cp;
  for (std::size_t b = 0; b < 2 + grid.size(); ++b)
  {
    cp.push_back(std::make_unique<ParticleSet>(particles[b < 2 ? b : grid[b - 2].sample]));
    if (b >= 2)
      for (int d = 0; d < 3; ++d) cp.back()->R[grid[b - 2].electron][d] = grid[b - 2].replacement_position[d];
    cp.back()->update();
  }
  PsiFormerComplexWF cpu("actual_ecp", initialized, *cp[0], source, clone_cap, twist);
  std::vector<std::unique_ptr<WaveFunctionComponent>> copies;
  RefVectorWithLeader<WaveFunctionComponent> components(cpu, {cpu});
  RefVectorWithLeader<ParticleSet> cpu_particles(*cp[0], {*cp[0]});
  for (std::size_t b = 1; b < cp.size(); ++b)
  {
    copies.push_back(cpu.makeClone(*cp[b])); components.push_back(*copies.back()); cpu_particles.push_back(*cp[b]);
  }
  evaluate(cpu, *cp[0]); evaluate(*copies[0], *cp[1]);
  std::array<C, 2> nonlocal_energy{};
  std::vector<C> weights(grid.size()), folded(cp.size());
  for (std::size_t q = 0; q < grid.size(); ++q)
  {
    const auto& point = grid[q];
    VirtualParticleSet vp(*cp[point.sample]);
    ParticleSet::PosType delta;
    for (int d = 0; d < 3; ++d) delta[d] = point.replacement_position[d] - cp[point.sample]->R[point.electron][d];
    vp.makeMoves(*cp[point.sample], point.electron, std::vector<ParticleSet::PosType>{delta});
    std::vector<WaveFunctionComponent::ValueType> ratio(1);
    components[point.sample].evaluateRatios(vp, ratio);
    close(ratio[0], point.complete_ratio, 3e-9);
    weights[q] = point.bare_weight * ratio[0]; nonlocal_energy[point.sample] += weights[q];
  }
  for (std::size_t b = 0; b < 2; ++b)
  {
    const auto observed = (b ? gpu1 : gpu).observeCapturedLocalEnergy(particles[b], *op, b, hamiltonians[b].getLocalPotential());
    const auto cpu_kinetic = dynamic_cast<PsiFormerComplexWF&>(components[b]).observeCoulombEnergy(*cp[b], 0).kinetic;
    close(observed.nonlocal, nonlocal_energy[b], 3e-9);
    close(inline_energies[b], nonlocal_energy[b], 3e-9);
    close(observed.local_energy, cpu_kinetic + local_baseline(particles[b]) + nonlocal_energy[b], 3e-9);
    close(observed.local_energy.real(), actual.local_energies[b], 3e-9);
    CHECK(observed.reported_real_potential == hamiltonians[b].getLocalPotential());
    CHECK(observed.grid_fingerprint == actual.grid_fingerprint); CHECK(observed.parameter_version == 0);
  }
  CHECK(std::abs(nonlocal_energy[0].imag()) + std::abs(nonlocal_energy[1].imag()) > 1e-9);
  auto oracle = cpu.makeBudgetedStreamingDerivativeOperator(components, cpu_particles, 19, 23, 7,
      cpu.streamingDerivativeRequiredBytes(cp.size()));
  auto score = channel; score.product = DerivativeProduct::SCORE_VJP; score.local_energy_term_mask = 0;
  score.coefficients.values = {folded.data(), folded.size()};
  for (auto adjoint : {DerivativeAdjoint::TRANSPOSE, DerivativeAdjoint::HERMITIAN})
  {
    std::fill(folded.begin(), folded.end(), C{});
    for (std::size_t q = 0; q < grid.size(); ++q)
    {
      folded[q + 2] = coefficients[grid[q].sample] * (adjoint == DerivativeAdjoint::HERMITIAN ? std::conj(weights[q]) : weights[q]);
      folded[grid[q].sample] -= folded[q + 2];
    }
    ComplexStreamingSink expected, output;
    oracle->applyVJPs({&score, 1}, adjoint, expected); op->applyVJPs({&channel, 1}, adjoint, output);
    for (std::size_t i = 0; i < direction.size(); ++i) close(output.values[0][i], expected.values[0][i], 3e-9);
    if (adjoint == DerivativeAdjoint::TRANSPOSE)
      for (std::size_t i = 0; i < direction.size(); ++i)
        close(inline_response.weighted_vjp[i], expected.values[0][i], 3e-9);
    ComplexSampleSink jvp;
    op->applyLocalEnergyJVP({schema, 0, {direction.data(), direction.size()}}, channel.local_energy_term_mask, jvp);
    C lhs{}, rhs{};
    for (std::size_t b = 0; b < 2; ++b)
      lhs += (adjoint == DerivativeAdjoint::HERMITIAN ? std::conj(coefficients[b]) : coefficients[b]) * jvp.values[b];
    for (std::size_t i = 0; i < direction.size(); ++i)
      rhs += (adjoint == DerivativeAdjoint::HERMITIAN ? std::conj(output.values[0][i]) : output.values[0][i]) * direction[i];
    close(lhs, rhs, 3e-9);
  }
  CHECK(gpu.snapshotParameters().values == accepted.values);
  // This explicit numerical route is NOT used by the native REAL-energy line-search selector.
  CHECK_THROWS(gpu.makeBudgetedFixedGridEnergyReference(p, *op, 0, {}, reference_cap));
  CHECK_THROWS(gpu.makeBudgetedFixedGridEnergyReference(p, *op, 0, actual.native_capture, reference_cap - 1));
  CHECK_THROWS(gpu.makeBudgetedFixedGridEnergyReference(p, *op, 2, actual.native_capture, reference_cap));
  auto fixed_grid = gpu.makeBudgetedFixedGridEnergyReference(p, *op, 0, actual.native_capture, reference_cap);
  CHECK_THROWS(gpu1.makeBudgetedFixedGridEnergyReference(p1, *op, 0, actual.native_capture, reference_cap));
  PsiFormerComplexWF private_cpu("actual_ecp", initialized, private_cpu_p, source, clone_cap, twist);
  const auto private_publication = private_gpu.gpuParameterPublicationRequirements();
  const auto real_leaf = std::find_if(initialized.layouts.begin(), initialized.layouts.end(),
      [](const auto& leaf) { return leaf.name == "w"; });
  const auto imag_leaf = std::find_if(initialized.layouts.begin(), initialized.layouts.end(),
      [](const auto& leaf) { return leaf.name == "w_imag"; });
  REQUIRE(real_leaf != initialized.layouts.end()); REQUIRE(imag_leaf != initialized.layouts.end());
  PsiFormerComplexWF::FixedGridCandidateEnergy candidate_energy;
  const auto source_g = p.G; const auto source_l = p.L; const auto source_positions = p.R;
  for (std::size_t step = 0; step < 2; ++step)
  {
    auto candidate = private_gpu.snapshotParameters();
    candidate.values[real_leaf->begin] += .013 * (step + 1);
    candidate.values[imag_leaf->begin] -= .017 * (step + 1);
    CHECK(private_gpu.publishBudgetedGpuParameters(candidate, candidate.version,
        private_publication.device_bytes, private_publication.host_bytes) == candidate.version + 1);
    CHECK(private_cpu.publishParameters(candidate, candidate.version) == candidate.version + 1);
    if (step)
    {
      CHECK_THROWS(gpu.validateFixedGridCandidateEnergy(candidate_energy, private_gpu));
      candidate_energy = {}; // Retire its pinned old image BEFORE measuring this generation's warm work.
    }
    evaluate(private_gpu, private_p); evaluate(private_cpu, private_cpu_p);
    C expected_nonlocal{};
    for (const auto& point : grid)
      if (point.sample == 0)
      {
        VirtualParticleSet virtuals(private_cpu_p);
        ParticleSet::PosType absolute;
        for (int d = 0; d < 3; ++d) absolute[d] = point.replacement_position[d];
        virtuals.makeMovesAbsolute(private_cpu_p, point.electron, {&absolute, 1});
        std::vector<WaveFunctionComponent::ValueType> ratios(1);
        private_cpu.evaluateRatios(virtuals, ratios);
        expected_nonlocal += point.bare_weight * ratios[0];
      }
    const auto kinetic = private_cpu.observeCoulombEnergy(private_cpu_p, 0).kinetic;
    const auto warm = metrics.snapshot(); const auto storage = private_gpu.numericStorage();
    candidate_energy = gpu.evaluateFixedGridCandidateEnergy(fixed_grid, private_gpu, private_p);
    gpu.validateFixedGridCandidateEnergy(candidate_energy, private_gpu);
    close(candidate_energy.kinetic, kinetic, 3e-9);
    close(candidate_energy.nonlocal, expected_nonlocal, 3e-9);
    close(candidate_energy.local_energy, kinetic + local_baseline(private_p) + expected_nonlocal, 3e-9);
    CHECK(candidate_energy.source_version == 0); CHECK(candidate_energy.candidate_version == step + 1);
    CHECK(candidate_energy.grid_fingerprint == actual.grid_fingerprint);
    close(gpu.evaluateFixedGridCandidateEnergy(fixed_grid, private_gpu, private_p).local_energy,
          candidate_energy.local_energy, 3e-9);
    CHECK(metrics.snapshot().allocation_calls == warm.allocation_calls);
    CHECK(metrics.snapshot().live_bytes == warm.live_bytes);
    CHECK(private_gpu.numericStorage().fingerprint == storage.fingerprint);
    const auto before_refusal = metrics.snapshot();
    CHECK_THROWS(gpu.evaluateFixedGridCandidateEnergy(fixed_grid, gpu1, p1));
    CHECK_THROWS(gpu1.evaluateFixedGridCandidateEnergy(fixed_grid, private_gpu, private_p));
    private_p.R[0][0] += .01; private_p.update();
    CHECK_THROWS(gpu.evaluateFixedGridCandidateEnergy(fixed_grid, private_gpu, private_p));
    private_p.R = source_positions; private_p.update();
    CHECK(metrics.snapshot().uploaded_bytes == before_refusal.uploaded_bytes);
    CHECK(metrics.snapshot().downloaded_bytes == before_refusal.downloaded_bytes);
    CHECK(gpu.snapshotParameters().values == accepted.values);
    for (int e = 0; e < p.getTotalNum(); ++e)
    {
      close(p.L[e], source_l[e]);
      for (int d = 0; d < 3; ++d)
      {
        CHECK(p.R[e][d] == source_positions[e][d]);
        close(p.G[e][d], source_g[e][d]);
      }
    }
  }
  const int private_mass_attribute = private_p.getSpeciesSet().addAttribute("mass");
  private_p.getSpeciesSet()(private_mass_attribute, 0) = 2; private_p.resetGroups();
  CHECK_THROWS(gpu.evaluateFixedGridCandidateEnergy(fixed_grid, private_gpu, private_p));
  private_p.getSpeciesSet()(private_mass_attribute, 0) = 1; private_p.resetGroups();
  evaluate(private_gpu, private_p);
  gpu.validateFixedGridCandidateEnergy(candidate_energy, private_gpu);
  // Preserve VALUE node semantics: coincident same-spin virtual rows are valid ratio0, not skipped work.
  VirtualParticleSet coincident(private_p);
  coincident.makeMovesAbsolute(private_p, 0, {&private_p.R[1], 1});
  std::vector<WaveFunctionComponent::ValueType> zero_ratio(1);
  private_gpu.evaluateRatios(coincident, zero_ratio);
  close(zero_ratio[0], C{}, 3e-9);
  const auto good_private = private_gpu.snapshotParameters();
  auto zero_private = good_private;
  for (const auto& leaf : initialized.layouts)
    if (leaf.module.find("Backflow") != std::string::npos)
      std::fill(zero_private.values.begin() + leaf.begin, zero_private.values.begin() + leaf.end, 0.);
  private_gpu.publishBudgetedGpuParameters(zero_private, zero_private.version,
      private_publication.device_bytes, private_publication.host_bytes);
  CHECK_THROWS(evaluate(private_gpu, private_p)); // A zero accepted denominator cannot publish FULL state.
  CHECK_THROWS(gpu.evaluateFixedGridCandidateEnergy(fixed_grid, private_gpu, private_p));
  auto recovered_private = good_private; recovered_private.version = zero_private.version + 1;
  private_gpu.publishBudgetedGpuParameters(recovered_private, recovered_private.version,
      private_publication.device_bytes, private_publication.host_bytes);
  evaluate(private_gpu, private_p);
  candidate_energy = gpu.evaluateFixedGridCandidateEnergy(fixed_grid, private_gpu, private_p);
  gpu.validateFixedGridCandidateEnergy(candidate_energy, private_gpu);
  std::printf("COMPLEX_FIXED_GRID_CANDIDATE PASS actual_native_seal=1 private_origin=1 two_versions=1 CPU_same_grid=1 no_Re_doublecount=1 warm_allocations=0 stale_foreign=1 source_unchanged=1\n");
  // Parameters-only PP import is fresh Gamma training state, not a sampler/PP restart.
  // Reuse this mathematical fixture's ACTUAL virtual positions/bare weights, but
  // recompute Gamma ratios: its original nonzero-twist ratios are not transferable.
  ParticleSet gamma_p(p), loaded_p(p);
  gamma_p.setTwist(ParticleSet::PosType{}); loaded_p.setTwist(ParticleSet::PosType{});
  PsiFormerComplexWF exported("pp_import", initialized, gamma_p, source, clone_cap,
      {}, 64 * 1024 * 1024, {}, "pseudopotential");
  auto changed = exported.snapshotParameters();
  changed.values.back() += .002;
  const auto publication = exported.gpuParameterPublicationRequirements();
  REQUIRE(exported.publishBudgetedGpuParameters(changed, 0, publication.device_bytes, publication.host_bytes) == 1);
  const auto export_state = exported.snapshotParameters();
  const auto export_log = evaluate(exported, gamma_p);
  const auto export_g = gamma_p.G; const auto export_l = gamma_p.L;
  const auto path = "psiformer_pp_direct." + std::to_string(OHMMS::Controller->rank()) + ".h5";
  REQUIRE_FALSE(std::filesystem::exists(path)); // Keep evidence from a failed prior attempt.
  exported.exportModel(path);
  WaveFunctionComponentBuilder::PSetMap import_sources;
  import_sources.emplace("ion0", std::make_unique<ParticleSet>(source));
  PsiFormerWaveFunctionBuilder importer(OHMMS::Controller, loaded_p, import_sources);
  auto load = [&](const std::string& controls, const std::string& kind = "pseudopotential") {
    Libxml2Document xml;
    REQUIRE(xml.parseFromString("<psiformer name='pp_import' parameters='" + path +
        "' source='ion0' system='" + kind + "' feature_policy='periodic_torus_v1' "
        "optimize='no' max_clone_numeric_bytes='8388608' " + controls + "/>"));
    return importer.buildComponent(xml.getRoot());
  };
  auto cpu_loaded_base = load("gpu='no' twist='0 0 0'");
  auto* cpu_loaded = dynamic_cast<PsiFormerComplexWF*>(cpu_loaded_base.get());
  REQUIRE(cpu_loaded != nullptr);
  CHECK_FALSE(cpu_loaded->gpuInferenceEnabled());
  CHECK(cpu_loaded->isPseudopotentialSystem());
  CHECK(cpu_loaded->snapshotParameters().version == 0);
  CHECK(cpu_loaded->snapshotParameters().values == export_state.values);
  close(evaluate(*cpu_loaded, loaded_p), export_log, 3e-9);
  for (int e = 0; e < loaded_p.getTotalNum(); ++e)
  {
    close(loaded_p.L[e], export_l[e], 3e-9);
    for (int d = 0; d < 3; ++d) close(loaded_p.G[e][d], export_g[e][d], 3e-9);
  }
  CHECK_THROWS_AS(load("gpu='no' twist='.1 0 0'"), std::invalid_argument);
#ifdef QMC_CUDA2HIP
  const std::string import_gpu = "gpu='hip' ";
#else
  const std::string import_gpu = "gpu='cuda' ";
#endif
  CHECK_THROWS_AS(load(import_gpu + "max_gpu_numeric_bytes='67108864' twist='.1 0 0'"), std::invalid_argument);
  CHECK_THROWS_AS(load(import_gpu + "max_gpu_numeric_bytes='67108864' twist='0 0 0'", "all_electron"), std::invalid_argument);
  auto loaded_base = load(import_gpu + "max_gpu_numeric_bytes='67108864' twist='0 0 0'");
  auto* loaded = dynamic_cast<PsiFormerComplexWF*>(loaded_base.get());
  REQUIRE(loaded != nullptr); REQUIRE(loaded->gpuInferenceEnabled());
  CHECK(loaded->isPseudopotentialSystem());
  CHECK(export_state.version == 1); CHECK(loaded->snapshotParameters().version == 0);
  CHECK(loaded->snapshotParameters().values == export_state.values);
  close(evaluate(*loaded, loaded_p), export_log, 3e-9);
  for (int e = 0; e < loaded_p.getTotalNum(); ++e)
  {
    close(loaded_p.L[e], export_l[e], 3e-9);
    for (int d = 0; d < 3; ++d) close(loaded_p.G[e][d], export_g[e][d], 3e-9);
  }
  std::array<C, 2> imported_energy{
      exported.observeCoulombEnergy(gamma_p, 0).kinetic,
      loaded->observeCoulombEnergy(loaded_p, 0).kinetic};
  for (const auto& point : grid)
    if (point.sample == 0)
      for (std::size_t owner = 0; owner < 2; ++owner)
      {
        auto& target = owner ? loaded_p : gamma_p;
        VirtualParticleSet vp(target);
        ParticleSet::PosType replacement;
        for (int d = 0; d < 3; ++d) replacement[d] = point.replacement_position[d];
        vp.makeMovesAbsolute(target, point.electron, {&replacement, 1});
        std::vector<WaveFunctionComponent::ValueType> ratios(1);
        (owner ? *loaded : exported).evaluateRatios(vp, ratios);
        imported_energy[owner] += point.bare_weight * ratios[0];
      }
  // This fixture's complete Hamiltonian is T+NLPP; native LiH/Ewald proof is separate.
  close(imported_energy[1], imported_energy[0], 3e-9);
  CHECK(exported.snapshotParameters().version == export_state.version);
  CHECK(exported.snapshotParameters().values == export_state.values);
  CHECK(gpu.snapshotParameters().values == accepted.values);
  std::filesystem::remove(path);
  std::printf("COMPLEX_PP_MODEL_FILE PASS fresh_version0 exact_values Gamma_full_GL complete_same_grid_energy source_unchanged\n");
  const int mass_attribute = p.getSpeciesSet().addAttribute("mass");
  p.getSpeciesSet()(mass_attribute, 0) = 2; p.resetGroups();
  CHECK_THROWS(gpu.observeCapturedLocalEnergy(p, *op, 0, ham.getLocalPotential()));
  p.getSpeciesSet()(mass_attribute, 0) = 1; p.resetGroups();
  CHECK_THROWS(gpu.observeCapturedLocalEnergy(p1, *op, 1, ham1->getLocalPotential()));
  const auto budget = gpu.gpuParameterPublicationRequirements();
  gpu.publishBudgetedGpuParameters(accepted, 0, budget.device_bytes, budget.host_bytes);
  CHECK_THROWS(gpu.validateFixedGridCandidateEnergy(candidate_energy, private_gpu));
  CHECK_THROWS(gpu.evaluateFixedGridCandidateEnergy(fixed_grid, private_gpu, private_p));
  CHECK_THROWS(gpu.observeCapturedLocalEnergy(p, *op, 0, ham.getLocalPotential()));
  std::printf("COMPLEX_ACTUAL_ECP PASS actual_traversal=1 complex_energy=1 real_reporting=1 CPU_allP=1 duality=1 cap_abort_stale=1\n");
}
#endif
TEST_CASE("PsiFormer paired GPU raw orbital-MSE matches independent CPU objective and preserves accepted state",
          "[wavefunction][psiformer][complex_gpu_orbital_mse]")
{
#ifdef ENABLE_CUDA
  int devices = 0;
#ifdef QMC_CUDA2HIP
  const bool available = hipGetDeviceCount(&devices) == hipSuccess && devices;
#else
  const bool available = cudaGetDeviceCount(&devices) == cudaSuccess && devices;
#endif
  if (!available) { WARN("NOT_RUN: orbital-MSE needs a native GPU"); return; }
  ScopedComplexGpuControls controls;
  const SimulationCell simulation_cell(cell());
  auto source = ions(simulation_cell);
  for (const std::array<double, 3> twist : {std::array<double, 3>{0, 0, 0}, {.17, -.11, .07}})
  {
    auto p = electrons(simulation_cell), cpu_p = electrons(simulation_cell);
    PsiFormerComplexWF cpu("mse", parameters(), cpu_p, source, clone_cap, twist);
    PsiFormerComplexWF gpu("mse", parameters(), p, source, clone_cap, twist, 64 * 1024 * 1024);
    const auto accepted = evaluate(gpu, p);
    const auto accepted_energy = gpu.observeCoulombEnergy(p, 0).kinetic;
    const auto state = gpu.snapshotParameters();
    std::vector<C> target;
    cpu.evaluateOrbitalValues(cpu_p, target, state.version);
    for (std::size_t i = 0; i < target.size(); ++i) target[i] += C{.013 * std::sin(double(i)), -.021};
    cpu.prepareOrbitalMSE(cpu.orbitalMSEStorage().required_bytes);
    int ordinal = -1;
#ifdef QMC_CUDA2HIP
    REQUIRE(hipGetDevice(&ordinal) == hipSuccess);
#else
    REQUIRE(cudaGetDevice(&ordinal) == cudaSuccess);
#endif
    auto& metrics = ps::device::deviceMemoryMetrics(ordinal);
    const auto requirement = gpu.orbitalMSEStorage();
    const auto before = metrics.snapshot();
    CHECK_THROWS(gpu.prepareOrbitalMSE(requirement.required_bytes - 1));
    CHECK(metrics.snapshot().allocation_calls == before.allocation_calls);
    CHECK(gpu.orbitalMSEStorage().prepared_bytes == 0);
    gpu.prepareOrbitalMSE(requirement.required_bytes);
    const auto prepared = gpu.orbitalMSEStorage();
    CHECK(prepared.prepared_bytes == requirement.required_bytes);
    std::vector<C> raw, expected_raw;
    gpu.evaluateOrbitalValues(p, raw, state.version);
    cpu.evaluateOrbitalValues(cpu_p, expected_raw, state.version);
    for (std::size_t i = 0; i < raw.size(); ++i) close(raw[i], expected_raw[i], 2e-12);
    std::vector<double> gradient(state.values.size(), 71), expected(gradient.size());
    for (auto normalization : {wftrain::OrbitalLossNormalization::SPIN_SECTOR_SUM,
        wftrain::OrbitalLossNormalization::FULL_MATRIX_MEAN})
    {
      const auto warm = metrics.snapshot();
      const auto loss = gpu.evaluateOrbitalMSE(p, target, gradient, state.version, normalization);
      const auto expected_loss = cpu.evaluateOrbitalMSE(cpu_p, target, expected, state.version, normalization);
      CHECK(loss == Catch::Approx(expected_loss).margin(2e-12));
      for (std::size_t i = 0; i < gradient.size(); ++i)
        CHECK(gradient[i] == Catch::Approx(expected[i]).margin(2e-12));
      CHECK(metrics.snapshot().allocation_calls == warm.allocation_calls);
    }
    const auto good_gradient = gradient;
    auto bad = target; bad.back() = {std::numeric_limits<double>::quiet_NaN(), 0};
    CHECK_THROWS(gpu.evaluateOrbitalMSE(p, bad, gradient, state.version)); CHECK(gradient == good_gradient);
    CHECK_THROWS(gpu.evaluateOrbitalMSE(p, target, gradient, state.version + 1)); CHECK(gradient == good_gradient);
    const auto source_x = source.R[0][0]; source.R[0][0] += .01;
    CHECK_THROWS(gpu.evaluateOrbitalMSE(p, target, gradient, state.version)); CHECK(gradient == good_gradient);
    source.R[0][0] = source_x;
    CHECK(gpu.get_log_value() == accepted);
    close(gpu.observeCoulombEnergy(p, 0).kinetic, accepted_energy, 2e-12);
    CHECK(gpu.snapshotParameters().values == state.values);
    CHECK(gpu.orbitalMSEStorage().storage_fingerprint == prepared.storage_fingerprint);
    // Raw orbitals remain defined at a complete wavefunction node.
    const auto saved = p.R[1]; p.R[1] = p.R[0]; cpu_p.R[1] = cpu_p.R[0];
    gpu.evaluateOrbitalValues(p, raw, state.version); cpu.evaluateOrbitalValues(cpu_p, expected_raw, state.version);
    for (std::size_t i = 0; i < raw.size(); ++i) close(raw[i], expected_raw[i], 2e-12);
    p.R[1] = saved; cpu_p.R[1] = saved;
    auto candidate = state; candidate.values[1] += .001;
    const auto budget = gpu.gpuParameterPublicationRequirements();
    CHECK(gpu.publishBudgetedGpuParameters(candidate, state.version, budget.device_bytes, budget.host_bytes) == 1);
    CHECK(cpu.publishParameters(candidate, state.version) == 1);
    const auto new_loss = gpu.evaluateOrbitalMSE(p, target, gradient, 1);
    CHECK(new_loss == Catch::Approx(cpu.evaluateOrbitalMSE(cpu_p, target, expected, 1)).margin(2e-12));
    for (std::size_t i = 0; i < gradient.size(); ++i)
      CHECK(gradient[i] == Catch::Approx(expected[i]).margin(2e-12));
    CHECK(gpu.orbitalMSEStorage().storage_fingerprint == prepared.storage_fingerprint);
  }
#else
  WARN("NOT_RUN: orbital-MSE GPU provider requires a GPU build");
#endif
}
} // namespace qmcplusplus

namespace qmcplusplus
{
TEST_CASE("CPU paired energies preserve accepted ordinary nonlocal quadrature",
          "[wavefunction][psiformer][complex_cpu_ecp_energy]")
{
  using namespace wftrain;
  struct Point
  {
    std::size_t sample, electron;
    ParticleSet::PosType position;
    C bare, ratio;
  };
  class Audit final : public NonLocalECPDerivativeConsumer
  {
  public:
    /// Keep the production scalar reduction and an independent test-only accepted-point oracle.
    Audit(const StructuredParameterSchema& schema) : sum(schema, 0, 19, 23, 2, 2, 2 * sizeof(C)) {}

    /// Forward the exact pre-RNG identity check without any parameter output.
    void preflight(const NonLocalECPDerivativeContext& c, DerivativeArrayView<const VJPCoefficientChannel> v,
                   const ParameterReductionSink& s) const override { sum.preflight(c, v, s); }

    /// Bind both numerical and audit paths to one actual native grid.
    void begin(const NonLocalECPDerivativeContext& c, DerivativeArrayView<const VJPCoefficientChannel> v,
               ParameterReductionSink& s) override { sum.begin(c, v, s); }

    /// Retain test-only coordinates before the native transient buffers are reused.
    void consume(const NonLocalECPDerivativeTile& tile) override
    {
      for (std::size_t segment = 0; segment < tile.virtual_particles.segmentCount(); ++segment)
      {
        const auto slice = tile.virtual_particles.slice(segment);
        for (std::size_t q = 0; q < slice.size(); ++q)
        {
          const auto index = slice.flatOffset() + q;
          points.push_back({static_cast<std::size_t>(slice.walkerId()), static_cast<std::size_t>(slice.electronId()),
              slice.absolutePosition(q), tile.bare_weights[index], tile.complete_ratios[index]});
        }
      }
      sum.consume(tile);
    }

    /// Require complete native traversal in the production reduction.
    void end() override { sum.end(); }

    /// Poison the partial production result without publishing it.
    void abort() noexcept override { sum.abort(); }

    /// The production owner has no parameter-direction replay.
    std::unique_ptr<NonLocalECPJVPReplay> releaseJVPReplay() override { return sum.releaseJVPReplay(); }

    /// Exclude the deliberately test-only point bank from production storage diagnostics.
    StreamingDerivativeStorageDiagnostics storageDiagnostics() const override { return sum.storageDiagnostics(); }

    NonLocalEnergyAccumulator sum;
    std::vector<Point> points;
  };
  class NoDerivativeSink final : public ParameterReductionSink
  {
    /// A score-only energy capture must never produce a local-energy parameter derivative.
    void consume(std::size_t, const ParameterChunkConstView&) override
    { throw std::logic_error("Unexpected nonlocal parameter derivative"); }
  } sink;
  Lattice lattice;
  lattice.R = {6.6, 0, 0, 0, 6.6, 0, 0, 0, 6.6};
  lattice.BoxBConds = {true, true, true}; lattice.reset();
  const SimulationCell simulation_cell(lattice);
  std::array<C, 2> positive{};
  for (const double sign : {1.0, -1.0})
  {
    ParticleSet source(simulation_cell), p(simulation_cell);
    source.setName("ion0"); source.create({2});
    auto& ions = source.getSpeciesSet(); const int li = ions.addSpecies("Li");
    ions(ions.addAttribute("charge"), li) = 1;
    ions(ions.addAttribute("atomicnumber"), li) = 3;
    source.R[0] = {0, 0, 0}; source.R[1] = {3.3, 3.3, 3.3};
    source.resetGroups(); source.update();
    p.setName("e"); p.create({1, 1});
    auto& species = p.getSpeciesSet(); species.addSpecies("u"); species.addSpecies("d");
    const int mass = species.addAttribute("mass"); species(mass, 0) = species(mass, 1) = 1;
    p.R[0] = {.7, .4, .2}; p.R[1] = {2.8, 3.1, 3.5};
    p.resetGroups(); p.addTable(source); p.update();
    auto nonlocal = std::make_unique<NonLocalECPotential>(source, p, false, true);
    auto* nl0 = nonlocal.get();
    const char* selected = std::getenv("PSIFORMER_TEST_ECP_XML");
    const auto pp_path = selected ? std::filesystem::path(selected) :
        std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() /
        "tests/molecules/LiH_pp/Li.BFD.xml";
    ECPComponentBuilder builder("scalar_energy", OHMMS::Controller, 4, 1);
    REQUIRE(builder.read_pp_file(pp_path.string())); REQUIRE(builder.pp_nonloc);
    nonlocal->addComponent(0, std::move(builder.pp_nonloc));
    testing::TestNonLocalECPotential::setTileCapacity(*nonlocal, 5);
    p.update(); ParticleSet p1(p); p1.R[0][2] += .037; p1.update();
    ps::ExecutionEnvironment environment;
    environment.boundary = ps::BoundaryCondition::PERIODIC;
    environment.geometry_feature_policy = ps::GeometryFeaturePolicy::PERIODIC_TORUS_V1;
    environment.head_representation = ps::HeadRepresentation::PAIRED_REAL_COMPLEX;
    environment.amplitude_scalar_domain = ps::ScalarDomain::COMPLEX;
    environment.periodic_axes = {true, true, true};
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) environment.lattice_vectors[i][j] = lattice.R(i, j);
    auto leaves = fixture::makeLayout(2, 2, false, 1, 7);
    for (auto& leaf : leaves)
      for (auto& extent : leaf.shape)
        if (extent == 256) extent = 8;
        else if (extent == 32) extent = 4;
    const auto real_leaves = leaves;
    for (const auto& leaf : real_leaves)
      if (leaf.module.find("Backflow") != std::string::npos)
        leaves.push_back({leaf.module, "w_imag", leaf.shape});
    std::sort(leaves.begin(), leaves.end(), [](const auto& a, const auto& b) {
      return std::tie(a.module, a.name) < std::tie(b.module, b.name);
    });
    ps::InitializedPsiFormerParameters initialized;
    initialized.model_shape = {1, 1, 2, 2, 8, 2, 1};
    initialized.profile = ps::PSIFORMER_COMPLEX_PERIODIC_V1;
    initialized.seed = 17; initialized.environment = environment;
    initialized.values = fixture::makeParameters("h2", leaves);
    std::size_t cursor = 0;
    for (const auto& leaf : leaves)
    {
      const auto end = cursor + fixture::product(leaf.shape);
      initialized.layouts.push_back({leaf.module, leaf.name, leaf.shape, cursor, end});
      cursor = end;
    }
    // A real periodic part isolates the conjugate response of the common nonzero boundary twist.
    for (const auto& layout : initialized.layouts)
      if (layout.name == "w_imag") std::fill(initialized.values.begin() + layout.begin,
                                             initialized.values.begin() + layout.end, 0.0);
    const std::array<double, 3> twist{sign / 8, sign / 5, sign / 7};
    RuntimeOptions options; TrialWaveFunction trial(options);
    auto owner = std::make_unique<PsiFormerComplexWF>("cpu_ecp", initialized, p, source, clone_cap,
        twist, 0, "", "pseudopotential");
    auto& cpu = *owner; trial.addComponent(std::move(owner));
    auto trial1 = trial.makeClone(p1);
    QMCHamiltonian ham("operator_energy_oracle");
    ham.addOperator(std::make_unique<BareKineticEnergy>(p), "Kinetic", true);
    ham.addOperator(std::move(nonlocal), "NonLocalECP", true);
    auto local = std::make_unique<HarmonicExternalPotential>(p);
    local->mass = local->energy = local->length = 1; local->center = 0;
    ham.addOperator(std::move(local), "LocalBaseline", true); ham.addObservables(p);
    auto ham1 = ham.makeClone(p1, *trial1);
    auto& nl1 = dynamic_cast<NonLocalECPotential&>(*ham1->getComponent(1));
    FakeRandom<double> rng0, rng1; rng0.set_value(.371); rng1.set_value(.371);
    nl0->setRandomGenerator(&rng0); nl1.setRandomGenerator(&rng1);
    RefVectorWithLeader<TrialWaveFunction> trials(trial, {trial, *trial1});
    RefVectorWithLeader<ParticleSet> particles(p, {p, p1});
    RefVectorWithLeader<QMCHamiltonian> hamiltonians(ham, {ham, *ham1});
    ResourceCollection pr("cpu ECP particles"), wr("cpu ECP wavefunctions"), hr("cpu ECP H");
    p.createResource(pr); trial.createResource(wr); ham.createResource(hr);
    ResourceCollectionTeamLock<ParticleSet> pl(pr, particles);
    ResourceCollectionTeamLock<TrialWaveFunction> wl(wr, trials);
    ResourceCollectionTeamLock<QMCHamiltonian> hl(hr, hamiltonians);
    trial.evaluateLog(p); trial1->evaluateLog(p1);
    const auto snapshot = cpu.snapshotParameters();
    // Orbital fitting uses only the neural value/score owners, even for a physical PP component.
    const auto accepted_log = cpu.get_log_value();
    const auto accepted_positions = p.R;
    const auto numeric = cpu.numericStorage();
    const double unevaluated_energy = ham.getLocalEnergy();
    std::vector<C> orbital_values;
    cpu.evaluateOrbitalValues(p, orbital_values, snapshot.version);
    REQUIRE(orbital_values.size() == cpu.orbitalMSEChannelCount() * 4);
    CHECK(cpu.orbitalMSEStorage().prepared_bytes == 0);
    CHECK(cpu.numericStorage().allocated_bytes == numeric.allocated_bytes);
    std::vector<C> fixed_orbitals(orbital_values.size());
    double independent_loss = 0;
    for (std::size_t index = 0; index < fixed_orbitals.size(); ++index)
    {
      fixed_orbitals[index] = index % 4 == 0 || index % 4 == 3 ? C{.7, .2} : C{};
      independent_loss += std::norm(orbital_values[index] - fixed_orbitals[index]) /
          (cpu.orbitalMSEChannelCount() * 2);
    }
    const auto mse_bytes = cpu.orbitalMSEStorage().required_bytes;
    REQUIRE(mse_bytes > 0);
    CHECK_THROWS(cpu.prepareOrbitalMSE(mse_bytes - 1));
    CHECK(cpu.orbitalMSEStorage().prepared_bytes == 0);
    cpu.prepareOrbitalMSE(mse_bytes);
    const auto prepared_numeric = cpu.numericStorage();
    std::vector<double> orbital_gradient(snapshot.values.size());
    const double orbital_loss = cpu.evaluateOrbitalMSE(p, fixed_orbitals, orbital_gradient, snapshot.version);
    CHECK(orbital_loss == Catch::Approx(independent_loss).epsilon(1e-12));
    CHECK(ps::determinant::isFiniteReal(orbital_loss));
    CHECK(std::all_of(orbital_gradient.begin(), orbital_gradient.end(),
                      [](double value) { return ps::determinant::isFiniteReal(value); }));
    CHECK(std::any_of(orbital_gradient.begin(), orbital_gradient.end(),
                      [](double value) { return std::abs(value) > 1e-8; }));
    CHECK(cpu.orbitalMSEStorage().prepared_bytes == mse_bytes);
    CHECK(cpu.numericStorage().allocated_bytes == prepared_numeric.allocated_bytes);
    CHECK(cpu.numericStorage().fingerprint == prepared_numeric.fingerprint);
    CHECK(cpu.get_log_value() == accepted_log);
    for (std::size_t electron = 0; electron < p.R.size(); ++electron)
      CHECK(p.R[electron] == accepted_positions[electron]);
    CHECK(ham.getLocalEnergy() == unevaluated_energy);
    CHECK(cpu.snapshotParameters().version == snapshot.version);
    CHECK(cpu.snapshotParameters().values == snapshot.values);
    const auto storage = NonLocalEnergyAccumulator::requiredBytes(2);
    Audit audit(cpu.parameterSchema());
    std::vector<C> ones(2, {1, 0});
    VJPCoefficientChannel channel{"energy", DerivativeProduct::LOCAL_ENERGY_VJP,
        {cpu.parameterSchema().providerId(), cpu.parameterSchema().fingerprint(), 0, 19, 23, {ones.data(), 2}},
        localEnergyTermBit(LocalEnergyTerm::NONLOCAL_ECP)};
    const auto result = QMCHamiltonian::mw_evaluateWithStreamingNonLocalECPDerivatives(
        hamiltonians, trials, particles, audit, {&channel, 1}, sink);
    REQUIRE(audit.points.size() > 5);
    CHECK(audit.sum.gridFingerprint() == result.grid_fingerprint);
    CHECK(audit.sum.storageDiagnostics().retained_numeric_bytes == storage);
    CHECK(sink.state() == DerivativeSinkState::IDLE);
    std::array<C, 2> independent{};
    for (const auto& point : audit.points)
    {
      auto& original = particles[point.sample];
      ParticleSet private_p(original);
      private_p.R[point.electron] = point.position; private_p.update();
      auto& component = *trials[point.sample].getOrbitals()[0];
      auto private_wf = component.makeClone(private_p);
      ParticleSet accepted_p(original);
      auto accepted_wf = component.makeClone(accepted_p);
      const auto ratio = std::exp(evaluate(*private_wf, private_p) - evaluate(*accepted_wf, accepted_p));
      close(ratio, point.ratio, 3e-10);
      independent[point.sample] += point.bare * ratio;
    }
    for (std::size_t sample = 0; sample < 2; ++sample)
    {
      close(audit.sum.energies()[sample], independent[sample], 3e-10);
      const auto& nl = dynamic_cast<const NonLocalECPotential&>(*hamiltonians[sample].getComponent(1));
      CHECK(independent[sample].real() == Catch::Approx(nl.getValue()).margin(3e-10));
      auto& component = dynamic_cast<PsiFormerComplexWF&>(*trials[sample].getOrbitals()[0]);
      const auto observation = component.observeCoulombEnergy(particles[sample], hamiltonians[sample].getLocalPotential());
      const C full = observation.local_energy + C{0, audit.sum.energies()[sample].imag()};
      CHECK(full.real() == Catch::Approx(result.local_energies[sample]).margin(3e-10));
      double baseline = 0;
      for (const auto& position : particles[sample].R) baseline += .5 * dot(position, position);
      close(full, observation.kinetic + baseline + independent[sample], 3e-10);
      if (sign > 0) positive[sample] = independent[sample];
      else close(independent[sample], std::conj(positive[sample]), 3e-10);
    }
    CHECK(std::abs(independent[0].imag()) + std::abs(independent[1].imag()) > 1e-7);
    CHECK(cpu.snapshotParameters().values == snapshot.values);
  }
}
} // namespace qmcplusplus
