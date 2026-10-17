//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file test_psiformer_complex_wf.cpp
 * @brief Native-free scalar component tests; XML admission is deliberately unchanged.
 */
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#include "Particle/ParticleSet.h"
#include "Particle/MCMultiParticleMoves.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexWF.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerInitialization.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWF.h"
#include "QMCHamiltonians/BareKineticEnergy.h"
#include "QMCDrivers/WFTrain/PreparedEnergyGradientProducer.h"
#include "QMCDrivers/WFTrain/EnergyGradientAccumulator.h"
#include "QMCDrivers/WFTrain/DistributedParameterReduction.h"
#include "QMCDrivers/WFTrain/MatrixFreeStochasticReconfiguration.h"
#include "QMCDrivers/WFTrain/TrainingCheckpoint.h"
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "Utilities/RuntimeOptions.h"
#include "OhmmsData/Libxml2Doc.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWaveFunctionBuilder.h"
#include "QMCHamiltonians/HamiltonianFactory.h"
#include "Particle/VirtualParticleSet.h"
#include "Utilities/ResourceCollection.h"
#include "psiformer_test_utils.h"
#ifdef ENABLE_CUDA
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceComplexSpatial.h"
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceComplexProducts.h"
#endif
#include <algorithm>
#include <chrono>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iterator>
#include <limits>
#include <optional>
#include <system_error>
#include <tuple>
#include <vector>

namespace qmcplusplus
{
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
      "PSIFORMER_GPU_COMPLEX_TILE_CAPACITY", "PSIFORMER_GPU_ACTIVE_INFERENCE_CROWDS"};
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
ps::InitializedPsiFormerParameters parameters(bool paired = true, bool zero_imaginary = false)
{
  auto leaves = fixture::makeLayout(4, 2, true, 2, 7);
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
  result.model_shape = {2, 2, 2, 2, 8, 2, 2};
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
  for (int d = 0; d < 3; ++d)
    CHECK(p.getTwist()[d] == 0);
  auto& source = *pool.at("ion0");
  const int valence = source.getSpeciesSet().findAttribute("valence");
  source.getSpeciesSet()(valence, 0) = 2;
  CHECK_THROWS(build(complexXml())); // ECP effective charges cannot masquerade as AE metadata
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
} // namespace

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
    const auto owner_bytes = dev::estimateComplexValueCost(plan, 2).device_bytes + dev::estimateComplexSpatialCost(plan, 2).device_bytes;
    const auto before_rejection = metrics.snapshot();
    CHECK_THROWS_AS(PsiFormerComplexWF("pf", initialized, p0, source, clone_cap, twist, owner_bytes - 1), std::length_error);
    CHECK(metrics.snapshot().allocation_calls == before_rejection.allocation_calls);
    controls.target(1, owner_bytes, 1000000000);
    CHECK_THROWS_AS(PsiFormerComplexWF("pf", initialized, p0, source, clone_cap, twist, owner_bytes), std::invalid_argument);
    for (const auto* name : {"PSIFORMER_GPU_VGL_TARGETS", "PSIFORMER_GPU_TARGET_WORKSPACE_BYTES", "PSIFORMER_GPU_TARGET_SCALAR_WORK_BUDGET"})
      ::unsetenv(name);

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
} // namespace qmcplusplus
