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
#include "psiformer_test_utils.h"
#ifdef ENABLE_CUDA
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceComplexSpatial.h"
#endif
#include <algorithm>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
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
      "PSIFORMER_GPU_RESIDENT_ENERGY_SR_BYTES", "PSIFORMER_GPU_RESIDENT_ENERGY_SR_HOST_BYTES", "PSIFORMER_GPU_RESIDENT_ENERGY_SR_REUSE"};
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
ParticleSet electrons(const SimulationCell& simulation_cell)
{
  const auto geometry = fixture::makeGeometry("lih");
  ParticleSet result(simulation_cell);
  result.setName("e");
  result.create({2, 2});
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
  const SimulationCell simulation_cell(cell());
  auto p = electrons(simulation_cell), clone_p = electrons(simulation_cell);
  auto source = ions(simulation_cell);
  for (auto* target : {&p, &clone_p})
  {
    const int mass = target->getSpeciesSet().findAttribute("mass");
    target->getSpeciesSet()(mass, 0) = .7;
    target->getSpeciesSet()(mass, 1) = 1.9;
    target->resetGroups();
  }
  clone_p.R[1][2] += .037;
  clone_p.update();
  RuntimeOptions options;
  TrialWaveFunction trial(options);
  const std::array<double, 3> twist{.17, -.11, .07};
  auto owner = std::make_unique<PsiFormerComplexWF>("analysis", parameters(), p, source, clone_cap, twist,
                                                  64 * 1024 * 1024);
  auto& component = *owner;
  trial.addComponent(std::move(owner));
  auto clone_trial = trial.makeClone(clone_p);
  RefVectorWithLeader<TrialWaveFunction> trials(trial);
  trials.push_back(trial);
  trials.push_back(*clone_trial);
  RefVectorWithLeader<ParticleSet> particles(p);
  particles.push_back(p);
  particles.push_back(clone_p);
  const auto r0 = component.gpuDerivativeRequirements(0, 2, 3, 7);
  const auto r2 = component.gpuDerivativeRequirements(2, 2, 3, 7);
  std::printf("GPU derivative admission R0_device=%zu R0_host=%zu R2_device=%zu R2_host=%zu matrix_work=%zu\n",
      r0.device_bytes, r0.host_bytes, r2.device_bytes, r2.host_bytes, r2.matrix_work);
  CHECK(r2.device_bytes == r0.device_bytes);
  CHECK(r2.host_bytes == r0.host_bytes + 32 * 2 * 4);
  CHECK(component.gpuDerivativeRequirements(4, 2, 3, 7).host_bytes == r0.host_bytes + 32 * 4 * 4);
  auto& metrics = dev::deviceMemoryMetrics(ordinal);
  const auto before = metrics.snapshot();
  CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, 2, 3, 7,
      r2.device_bytes - 1, r2.host_bytes, r2.matrix_work));
  CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, 2, 3, 7,
      r2.device_bytes, r2.host_bytes - 1, r2.matrix_work));
  CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, 2, 3, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work - 1));
  const auto mass = p.getSpeciesSet().findAttribute("mass");
  p.getSpeciesSet()(mass, 0) = 0;
  p.resetGroups();
  CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, 2, 3, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work));
  p.getSpeciesSet()(mass, 0) = .7;
  p.resetGroups();
  p.setSpinor(true);
  CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, 2, 3, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work));
  p.setSpinor(false);
  const auto last_spin = p.spins[3];
  p.spins.resize(3);
  CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, 2, 3, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work));
  p.spins.resize(4);
  p.spins[3] = last_spin;
  const auto spin = p.spins[0];
  p.spins[0] = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, 2, 3, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work));
  p.spins[0] = spin;
  const int charge = source.getSpeciesSet().findAttribute("charge");
  const auto source_charge = source.getSpeciesSet()(charge, 0);
  const auto source_coordinate = source.R[0][0];
  for (int field = 0; field < 2; ++field)
  {
    auto& scalar = field ? source.R[0][0] : source.getSpeciesSet()(charge, 0);
    scalar = std::numeric_limits<double>::quiet_NaN();
    CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, 2, 3, 7,
        r2.device_bytes, r2.host_bytes, r2.matrix_work));
    scalar = field ? source_coordinate : source_charge;
  }
  CHECK(metrics.snapshot().allocation_calls == before.allocation_calls);
  auto stream = component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, 2, 3, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work);
  const auto diagnostic = stream->storageDiagnostics();
  CHECK(diagnostic.retained_numeric_bytes == r2.device_bytes + r2.host_bytes);
  CHECK(diagnostic.parameter_scratch_bytes == 144 * component.parameterSchema().parameterCount() + 16 * 7);
  CHECK(diagnostic.evaluator_workspace_bytes + diagnostic.parameter_scratch_bytes == diagnostic.shared_workspace_bytes);
  CHECK(diagnostic.shared_workspace_bytes == r0.device_bytes + r0.host_bytes);
  CHECK_FALSE(stream->capabilities().supports(DerivativeProduct::SCORE_JVP));
  CHECK_FALSE(stream->capabilities().supports(DerivativeAdjoint::HERMITIAN));
  CHECK(component.structuredParameterProvider() == nullptr);
  const auto snapshot = component.snapshotParameters();
  CHECK_THROWS(component.publishParameters(snapshot, snapshot.version));

  // The independently qualified CPU two-tape owner supplies complete complex rows, not Re-only forces.
  PsiFormerComplexWF cpu("analysis", parameters(), p, source, clone_cap, twist);
  auto cpu_clone = cpu.makeClone(clone_p);
  RefVectorWithLeader<WaveFunctionComponent> components(cpu);
  components.push_back(cpu);
  components.push_back(*cpu_clone);
  auto oracle = cpu.makeBudgetedStreamingDerivativeOperator(components, particles, 7, 11, 7,
      cpu.streamingDerivativeRequiredBytes(2));
  CHECK_THROWS(cpu.gpuDerivativeRequirements(2, 2, 3, 7));
  const auto& schema = stream->parameterSchema();
  std::array<std::vector<C>, 3> coefficients{{{{.3, -.2}, {-.4, .1}},
      {{.7, .15}, {.2, -.35}}, {{-.6, .4}, {.8, -.1}}}};
  std::array<VJPCoefficientChannel, 3> channels;
  for (std::size_t c = 0; c < channels.size(); ++c)
    channels[c] = {c == 0 ? "score0" : c == 1 ? "kinetic" : "score2",
        c == 1 ? DerivativeProduct::LOCAL_ENERGY_VJP : DerivativeProduct::SCORE_VJP,
        {schema.providerId(), schema.fingerprint(), stream->parameterVersion(), 7, 11,
          {coefficients[c].data(), 2}}, c == 1 ? localEnergyTermBit(LocalEnergyTerm::KINETIC) : 0};
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
  auto second = component.makeBudgetedGpuDerivativeOperator(trials, particles, 8, 13, 2, 3, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work);
  CHECK(second->storageDiagnostics().shared_workspace_identity == diagnostic.shared_workspace_identity);
  CHECK(second->storageDiagnostics().retained_numeric_bytes + diagnostic.retained_numeric_bytes -
      diagnostic.shared_workspace_bytes == r0.device_bytes + r0.host_bytes + 32 * 4 * 4);
  CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, 2, 2, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work));
  actual.reset();
  actual.fail_next = true;
  CHECK_THROWS(stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual));
  CHECK(actual.state() == DerivativeSinkState::POISONED);
  actual.reset();
  CHECK_NOTHROW(stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual));
  actual.reset();
  CHECK_THROWS(stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::HERMITIAN, actual));
  CHECK(actual.state() == DerivativeSinkState::IDLE);
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
  component.restore(0);
  p.rejectMove(0);
  actual.reset();
  CHECK_NOTHROW(stream->applyVJPs({channels.data(), 3}, DerivativeAdjoint::TRANSPOSE, actual));
  trial.addComponent(component.makeClone(p));
  CHECK_THROWS(component.makeBudgetedGpuDerivativeOperator(trials, particles, 7, 11, 2, 3, 7,
      r2.device_bytes, r2.host_bytes, r2.matrix_work));
}
#endif
} // namespace qmcplusplus
