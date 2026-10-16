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
#include "QMCWaveFunctions/TrialWaveFunction.h"
#include "Utilities/RuntimeOptions.h"
#include "OhmmsData/Libxml2Doc.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerWaveFunctionBuilder.h"
#include "QMCHamiltonians/HamiltonianFactory.h"
#include "Particle/VirtualParticleSet.h"
#include "psiformer_test_utils.h"
#include <algorithm>
#include <complex>
#include <limits>
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
  CHECK(complex.structuredParameterProvider() == nullptr);
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
  for (const char* request : {"gpu='auto'", "gpu='cuda'", "optimize='yes'",
                             "precision='fp32_dense_fp64_islands'"})
    CHECK_THROWS(build(complexXml(request)));
  CHECK_THROWS(build(complexXml("", "")));
  CHECK_THROWS(build(complexXml("", "0")));
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
  CHECK(component.structuredParameterProvider() == nullptr); // live training discovery stays closed
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
  CHECK(component.structuredParameterProvider() == nullptr);
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
} // namespace qmcplusplus
