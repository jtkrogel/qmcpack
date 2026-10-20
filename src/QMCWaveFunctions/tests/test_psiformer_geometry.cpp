//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "QMCWaveFunctions/PsiFormer/PsiFormerGeometry.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerPeriodicFeatures.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerPeriodicGeometry.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerPeriodicPairFeatures.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

/// Check one scalar against a tight absolute and relative tolerance.
void checkClose(double actual, double expected, double tolerance = 2e-14)
{
  CHECK(actual == Catch::Approx(expected).epsilon(tolerance).margin(tolerance));
}

/// Test positive infinity without fast-math-sensitive classification builtins.
bool isPositiveInfinity(double value)
{
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits == 0x7ff0000000000000ULL;
}

/// Classify an IEEE-754 binary64 scalar independently of fast-math assumptions.
bool isFinite(double value)
{
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

/// Require every softened radial value and derivative to remain finite.
void checkFinite(const pf::SoftenedRadialFactors& factors)
{
  CHECK(isFinite(factors.log1p_radius));
  CHECK(isFinite(factors.log1p_over_radius));
  CHECK(isFinite(factors.log1p_first));
  CHECK(isFinite(factors.log1p_second));
  CHECK(isFinite(factors.log1p_over_radius_first));
  CHECK(isFinite(factors.log1p_over_radius_second));
}

/// Compare radial-factor tables exactly around a rejected transaction.
void checkRadialFactors(
    const std::vector<pf::SoftenedRadialFactors>& actual,
    const std::vector<pf::SoftenedRadialFactors>& expected)
{
  REQUIRE(actual.size() == expected.size());
  for (std::size_t pair = 0; pair < actual.size(); ++pair)
  {
    CHECK(actual[pair].log1p_radius == expected[pair].log1p_radius);
    CHECK(actual[pair].log1p_over_radius ==
          expected[pair].log1p_over_radius);
    CHECK(actual[pair].log1p_first == expected[pair].log1p_first);
    CHECK(actual[pair].log1p_second == expected[pair].log1p_second);
    CHECK(actual[pair].log1p_over_radius_first ==
          expected[pair].log1p_over_radius_first);
    CHECK(actual[pair].log1p_over_radius_second ==
          expected[pair].log1p_over_radius_second);
  }
}

/// Compare the complete accepted cache state around a rejected update.
void checkGeometryState(const pf::PsiFormerGeometryCache& cache,
                        const std::vector<pf::GeometryPosition>& electrons,
                        const std::vector<pf::GeometryPosition>& en_displacements,
                        const std::vector<double>& en_distances,
                        const std::vector<double>& en_inverse_distances,
                        const std::vector<pf::SoftenedRadialFactors>& en_radial,
                        const std::vector<pf::GeometryPosition>& ee_displacements,
                        const std::vector<double>& ee_distances,
                        const std::vector<double>& ee_inverse_distances,
                        const std::vector<pf::SoftenedRadialFactors>& ee_radial,
                        std::size_t generation)
{
  CHECK(cache.valid());
  CHECK(cache.generation() == generation);
  CHECK(cache.electrons() == electrons);
  CHECK(cache.electronNucleusPairs().displacements() == en_displacements);
  CHECK(cache.electronNucleusPairs().distances() == en_distances);
  CHECK(cache.electronNucleusPairs().inverseDistances() ==
        en_inverse_distances);
  checkRadialFactors(cache.electronNucleusPairs().softenedRadialFactors(),
                     en_radial);
  CHECK(cache.electronElectronPairs().displacements() == ee_displacements);
  CHECK(cache.electronElectronPairs().distances() == ee_distances);
  CHECK(cache.electronElectronPairs().inverseDistances() ==
        ee_inverse_distances);
  checkRadialFactors(cache.electronElectronPairs().softenedRadialFactors(),
                     ee_radial);
}

} // namespace

TEST_CASE("PsiFormer neutral geometry views", "[wavefunction][psiformer]")
{
  const std::array<double, 9> interleaved{1, 2, 3, 4, 5, 6, 7, 8, 9};
  const pf::GeometryPositionView aos = pf::GeometryPositionView::interleaved(interleaved.data(), 3);
  CHECK(aos.size() == 3);
  CHECK(aos.position(1) == pf::GeometryPosition{4, 5, 6});

  const std::array<double, 3> x{1, 4, 7};
  const std::array<double, 3> y{2, 5, 8};
  const std::array<double, 3> z{3, 6, 9};
  const pf::GeometryPositionView soa = pf::GeometryPositionView::components(x.data(), y.data(), z.data(), 3);
  CHECK(soa.position(2) == pf::GeometryPosition{7, 8, 9});

  CHECK_THROWS_AS(pf::GeometryPositionView::interleaved(interleaved.data(), 3, 2), std::invalid_argument);
  CHECK_THROWS_AS(aos(3, 0), std::out_of_range);
  CHECK_THROWS_AS(aos(0, 3), std::out_of_range);
}

TEST_CASE("PsiFormer softened radial factors remain stable near coalescence", "[wavefunction][psiformer]")
{
  const pf::SoftenedRadialFactors at_origin = pf::evaluateSoftenedRadialFactors(0);
  checkClose(at_origin.log1p_radius, 0);
  checkClose(at_origin.log1p_over_radius, 1);
  checkClose(at_origin.log1p_over_radius_first, -0.5);
  checkClose(at_origin.log1p_over_radius_second, 2.0 / 3.0);

  const double radius = 1.0e-12;
  const pf::SoftenedRadialFactors near_origin = pf::evaluateSoftenedRadialFactors(radius);
  checkClose(near_origin.log1p_over_radius, 1 - radius / 2, 2e-15);
  checkClose(near_origin.log1p_over_radius_first, -0.5 + 2 * radius / 3, 2e-15);
  checkClose(near_origin.log1p_over_radius_second, 2.0 / 3.0 - 1.5 * radius, 2e-15);

  const double ordinary_radius = 0.75;
  const pf::SoftenedRadialFactors ordinary = pf::evaluateSoftenedRadialFactors(ordinary_radius);
  checkClose(ordinary.log1p_over_radius, std::log1p(ordinary_radius) / ordinary_radius);
  checkClose(ordinary.log1p_first, 1 / (1 + ordinary_radius));
  checkClose(ordinary.log1p_second, -1 / ((1 + ordinary_radius) * (1 + ordinary_radius)));

  for (double adversarial_radius : {
           std::numeric_limits<double>::denorm_min(),
           std::numeric_limits<double>::min(),
           std::nextafter(1.0e-2, 0.0),
           1.0e-2,
           std::nextafter(1.0e-2,
                          std::numeric_limits<double>::infinity())})
    checkFinite(pf::evaluateSoftenedRadialFactors(adversarial_radius));

  const auto below_threshold = pf::evaluateSoftenedRadialFactors(
      std::nextafter(1.0e-2, 0.0));
  const auto above_threshold = pf::evaluateSoftenedRadialFactors(
      std::nextafter(1.0e-2, std::numeric_limits<double>::infinity()));
  checkClose(below_threshold.log1p_over_radius,
             above_threshold.log1p_over_radius, 2e-13);
  checkClose(below_threshold.log1p_over_radius_first,
             above_threshold.log1p_over_radius_first, 2e-11);
  checkClose(below_threshold.log1p_over_radius_second,
             above_threshold.log1p_over_radius_second, 2e-8);

  CHECK_THROWS_AS(pf::evaluateSoftenedRadialFactors(-0.1), std::invalid_argument);
  CHECK_THROWS_AS(pf::evaluateSoftenedRadialFactors(std::numeric_limits<double>::infinity()),
                  std::invalid_argument);
}

TEST_CASE("PsiFormer molecular geometry caches unique pairs and incidence", "[wavefunction][psiformer]")
{
  const std::array<double, 6> nuclei{-1, 0, 0, 2, 0, 0};
  pf::PsiFormerGeometryCache cache(3, pf::GeometryPositionView::interleaved(nuclei.data(), 2));
  CHECK_FALSE(cache.valid());
  CHECK(cache.electronNucleusPairs().size() == 6);
  CHECK(cache.electronElectronPairs().size() == 3);

  const std::array<double, 9> electrons{1, 2, 2, -1, 0, 0, 1, 2, -2};
  cache.update(pf::GeometryPositionView::interleaved(electrons.data(), 3));
  CHECK(cache.valid());
  CHECK(cache.generation() == 1);

  const auto& en = cache.electronNucleusPairs();
  CHECK(en.displacements()[cache.electronNucleusPairIndex(0, 0)] == pf::GeometryPosition{2, 2, 2});
  checkClose(en.distances()[cache.electronNucleusPairIndex(0, 0)], std::sqrt(12.0));
  CHECK(en.displacements()[cache.electronNucleusPairIndex(0, 1)] == pf::GeometryPosition{-1, 2, 2});
  checkClose(en.distances()[cache.electronNucleusPairIndex(0, 1)], 3);

  // Unique pair order is (0,1), (0,2), (1,2), and displacements always use
  // first-minus-second orientation.
  const std::vector<pf::ElectronPair> expected_pairs{{0, 1}, {0, 2}, {1, 2}};
  REQUIRE(cache.electronPairs().size() == expected_pairs.size());
  for (std::size_t pair = 0; pair < expected_pairs.size(); ++pair)
  {
    CHECK(cache.electronPairs()[pair].first == expected_pairs[pair].first);
    CHECK(cache.electronPairs()[pair].second == expected_pairs[pair].second);
  }
  CHECK(cache.electronElectronPairs().displacements()[0] == pf::GeometryPosition{2, 2, 2});
  CHECK(cache.electronElectronPairs().displacements()[1] == pf::GeometryPosition{0, 0, 4});
  CHECK(cache.electronElectronPairs().displacements()[2] == pf::GeometryPosition{-2, -2, 2});

  const std::vector<std::size_t> expected_offsets{0, 2, 4, 6};
  CHECK(cache.incidenceOffsets() == expected_offsets);
  for (std::size_t electron = 0; electron < cache.electronCount(); ++electron)
    for (std::size_t incidence = cache.incidenceOffsets()[electron];
         incidence < cache.incidenceOffsets()[electron + 1]; ++incidence)
    {
      const pf::ElectronPairIncidence& entry = cache.incidences()[incidence];
      const pf::ElectronPair& pair            = cache.electronPairs()[entry.pair_index];
      if (electron == pair.first)
      {
        CHECK(entry.other_electron == pair.second);
        CHECK(entry.displacement_sign == 1);
      }
      else
      {
        CHECK(electron == pair.second);
        CHECK(entry.other_electron == pair.first);
        CHECK(entry.displacement_sign == -1);
      }
    }

  // A coincident electron-nucleus pair keeps the softened feature finite while
  // exposing the physical 1/r singularity explicitly.
  const std::size_t coincident_pair = cache.electronNucleusPairIndex(1, 0);
  checkClose(en.softenedRadialFactors()[coincident_pair].log1p_over_radius, 1);
  CHECK(isPositiveInfinity(en.inverseDistances()[coincident_pair]));
}

TEST_CASE("PsiFormer geometry updates reuse storage", "[wavefunction][psiformer]")
{
  const std::array<double, 3> nuclei{0, 0, 0};
  pf::PsiFormerGeometryCache cache(2, pf::GeometryPositionView::interleaved(nuclei.data(), 1));
  const std::array<double, 6> electrons{1, 0, 0, -1, 0, 0};
  cache.update(pf::GeometryPositionView::interleaved(electrons.data(), 2));

  const pf::GeometryPosition* const en_storage = cache.electronNucleusPairs().displacements().data();
  const pf::GeometryPosition* const ee_storage = cache.electronElectronPairs().displacements().data();
  const pf::SoftenedRadialFactors* const radial_storage =
      cache.electronNucleusPairs().softenedRadialFactors().data();

  cache.updateElectron(0, pf::GeometryPosition{2, 0, 0});
  CHECK(cache.generation() == 2);
  CHECK(cache.electronNucleusPairs().displacements().data() == en_storage);
  CHECK(cache.electronElectronPairs().displacements().data() == ee_storage);
  CHECK(cache.electronNucleusPairs().softenedRadialFactors().data() == radial_storage);
  checkClose(cache.electronNucleusPairs().distances()[0], 2);
  checkClose(cache.electronElectronPairs().distances()[0], 3);

  CHECK_THROWS_AS(cache.update(pf::GeometryPositionView::interleaved(electrons.data(), 1)), std::invalid_argument);
  CHECK_THROWS_AS(cache.updateElectron(2, pf::GeometryPosition{0, 0, 0}), std::out_of_range);
  CHECK_THROWS_AS(cache.updateElectron(
                      0, pf::GeometryPosition{std::numeric_limits<double>::quiet_NaN(), 0, 0}),
                  std::invalid_argument);
}

TEST_CASE("PsiFormer geometry rejects derived overflow atomically",
          "[wavefunction][psiformer][hardening][atomic]")
{
  constexpr double largest = std::numeric_limits<double>::max();
  const std::array<double, 3> nuclei{-largest, 0, 0};
  pf::PsiFormerGeometryCache cache(
      2, pf::GeometryPositionView::interleaved(nuclei.data(), 1));
  const std::array<double, 6> initial{1, 0, 0, 2, 0, 0};
  cache.update(pf::GeometryPositionView::interleaved(initial.data(), 2));

  const auto electrons        = cache.electrons();
  const auto en_displacements = cache.electronNucleusPairs().displacements();
  const auto en_distances     = cache.electronNucleusPairs().distances();
  const auto en_inverse_distances =
      cache.electronNucleusPairs().inverseDistances();
  const auto en_radial =
      cache.electronNucleusPairs().softenedRadialFactors();
  const auto ee_displacements = cache.electronElectronPairs().displacements();
  const auto ee_distances     = cache.electronElectronPairs().distances();
  const auto ee_inverse_distances =
      cache.electronElectronPairs().inverseDistances();
  const auto ee_radial =
      cache.electronElectronPairs().softenedRadialFactors();
  const std::size_t generation = cache.generation();

  const std::array<double, 6> overflowing{largest, 0, 0, 2, 0, 0};
  CHECK_THROWS_AS(
      cache.update(pf::GeometryPositionView::interleaved(overflowing.data(), 2)),
      std::invalid_argument);
  checkGeometryState(cache, electrons, en_displacements, en_distances,
                     en_inverse_distances, en_radial, ee_displacements,
                     ee_distances, ee_inverse_distances, ee_radial, generation);

  CHECK_THROWS_AS(cache.updateElectron(0, pf::GeometryPosition{largest, 0, 0}),
                  std::invalid_argument);
  checkGeometryState(cache, electrons, en_displacements, en_distances,
                     en_inverse_distances, en_radial, ee_displacements,
                     ee_distances, ee_inverse_distances, ee_radial, generation);

  CHECK_NOTHROW(cache.updateElectron(0, pf::GeometryPosition{3, 0, 0}));
  CHECK(cache.generation() == generation + 1);
  CHECK(cache.electrons()[0] == pf::GeometryPosition{3, 0, 0});
}

TEST_CASE("PsiFormer periodic torus geometry is image invariant and differentiable",
          "[wavefunction][psiformer][periodic]")
{
  const pf::GeometryBoundary boundary{
      pf::GeometryBoundaryKind::PERIODIC,
      {{{5.0, 0.0, 0.0}, {0.7, 4.6, 0.0}, {-0.2, 0.4, 5.3}}},
      {{true, true, true}}};
  const std::array<double, 3> nuclei{0.35, -0.22, 0.41};
  const std::array<double, 3> electron{1.13, 0.87, -0.36};
  pf::PsiFormerGeometryCache cache(
      1, pf::GeometryPositionView::interleaved(nuclei.data(), 1), boundary);
  cache.update(pf::GeometryPositionView::interleaved(electron.data(), 1));

  const auto& pairs = cache.electronNucleusPairs();
  const pf::GeometryPosition q = pairs.displacements()[0];
  const pf::GeometryPosition p = pairs.complementaryDisplacements()[0];
  const double radius = pairs.distances()[0];

  // Moving the electron by two independent lattice vectors leaves the complete
  // periodic feature, not merely its scalar radius, unchanged.
  std::array<double, 3> image = electron;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    image[dimension] += boundary.lattice_vectors[0][dimension] -
        2 * boundary.lattice_vectors[2][dimension];
  cache.update(pf::GeometryPositionView::interleaved(image.data(), 1));
  for (std::size_t component = 0; component < 3; ++component)
  {
    checkClose(cache.electronNucleusPairs().displacements()[0][component], q[component], 2e-13);
    checkClose(cache.electronNucleusPairs().complementaryDisplacements()[0][component], p[component], 2e-13);
  }
  checkClose(cache.electronNucleusPairs().distances()[0], radius, 2e-13);

  // A common translation of ions and electrons is represented by a fresh cache
  // because nuclei are intentionally immutable inside one workspace.
  const pf::GeometryPosition translation{1.7, -0.9, 0.6};
  std::array<double, 3> shifted_nucleus{};
  std::array<double, 3> shifted_electron{};
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
  {
    shifted_nucleus[dimension] = nuclei[dimension] + translation[dimension];
    shifted_electron[dimension] = electron[dimension] + translation[dimension];
  }
  pf::PsiFormerGeometryCache shifted_cache(
      1, pf::GeometryPositionView::interleaved(shifted_nucleus.data(), 1), boundary);
  shifted_cache.update(
      pf::GeometryPositionView::interleaved(shifted_electron.data(), 1));
  for (std::size_t component = 0; component < 3; ++component)
  {
    checkClose(shifted_cache.electronNucleusPairs().displacements()[0][component], q[component], 2e-14);
    checkClose(shifted_cache.electronNucleusPairs().complementaryDisplacements()[0][component], p[component], 2e-14);
  }
  checkClose(shifted_cache.electronNucleusPairs().distances()[0], radius, 2e-14);

  // Independently difference all cached Cartesian first derivatives and traces.
  cache.update(pf::GeometryPositionView::interleaved(electron.data(), 1));
  const auto analytic_q_jacobian = pairs.displacementJacobians()[0];
  const auto analytic_p_jacobian = pairs.complementaryDisplacementJacobians()[0];
  const auto analytic_radius_gradient = pairs.distanceGradients()[0];
  const auto analytic_q_laplacian = pairs.displacementLaplacians()[0];
  const auto analytic_p_laplacian = pairs.complementaryDisplacementLaplacians()[0];
  const double analytic_radius_laplacian = pairs.distanceLaplacians()[0];
  constexpr double first_step = 2e-6;
  constexpr double second_step = 2e-4;
  pf::GeometryPosition finite_q_laplacian{};
  pf::GeometryPosition finite_p_laplacian{};
  double finite_radius_laplacian = 0;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
  {
    std::array<double, 3> plus = electron;
    std::array<double, 3> minus = electron;
    plus[dimension] += first_step;
    minus[dimension] -= first_step;
    cache.update(pf::GeometryPositionView::interleaved(plus.data(), 1));
    const pf::GeometryPosition q_plus = pairs.displacements()[0];
    const pf::GeometryPosition p_plus = pairs.complementaryDisplacements()[0];
    const double r_plus = pairs.distances()[0];
    cache.update(pf::GeometryPositionView::interleaved(minus.data(), 1));
    const pf::GeometryPosition q_minus = pairs.displacements()[0];
    const pf::GeometryPosition p_minus = pairs.complementaryDisplacements()[0];
    const double r_minus = pairs.distances()[0];
    for (std::size_t component = 0; component < 3; ++component)
    {
      checkClose((q_plus[component] - q_minus[component]) / (2 * first_step),
                 analytic_q_jacobian[component][dimension], 2e-9);
      checkClose((p_plus[component] - p_minus[component]) / (2 * first_step),
                 analytic_p_jacobian[component][dimension], 2e-9);
    }
    checkClose((r_plus - r_minus) / (2 * first_step),
               analytic_radius_gradient[dimension], 2e-9);

    plus = electron;
    minus = electron;
    plus[dimension] += second_step;
    minus[dimension] -= second_step;
    cache.update(pf::GeometryPositionView::interleaved(plus.data(), 1));
    const pf::GeometryPosition q_second_plus = pairs.displacements()[0];
    const pf::GeometryPosition p_second_plus = pairs.complementaryDisplacements()[0];
    const double r_second_plus = pairs.distances()[0];
    cache.update(pf::GeometryPositionView::interleaved(minus.data(), 1));
    const pf::GeometryPosition q_second_minus = pairs.displacements()[0];
    const pf::GeometryPosition p_second_minus = pairs.complementaryDisplacements()[0];
    const double r_second_minus = pairs.distances()[0];
    for (std::size_t component = 0; component < 3; ++component)
    {
      finite_q_laplacian[component] +=
          (q_second_plus[component] - 2 * q[component] + q_second_minus[component]) /
          (second_step * second_step);
      finite_p_laplacian[component] +=
          (p_second_plus[component] - 2 * p[component] + p_second_minus[component]) /
          (second_step * second_step);
    }
    finite_radius_laplacian +=
        (r_second_plus - 2 * radius + r_second_minus) / (second_step * second_step);
  }
  for (std::size_t component = 0; component < 3; ++component)
  {
    checkClose(finite_q_laplacian[component], analytic_q_laplacian[component], 2e-6);
    checkClose(finite_p_laplacian[component], analytic_p_laplacian[component], 2e-6);
  }
  checkClose(finite_radius_laplacian, analytic_radius_laplacian, 2e-6);
}

TEST_CASE("PsiFormer periodic geometry validates bulk cells", "[wavefunction][psiformer][periodic]")
{
  CHECK(pf::supportsGeometryBoundary(pf::GeometryBoundaryKind::OPEN));
  CHECK(pf::supportsGeometryBoundary(pf::GeometryBoundaryKind::PERIODIC));

  pf::GeometryBoundary periodic;
  periodic.kind          = pf::GeometryBoundaryKind::PERIODIC;
  periodic.periodic_axes = {true, true, true};
  periodic.lattice_vectors =
      {pf::GeometryPosition{8, 0, 0}, pf::GeometryPosition{0, 8, 0}, pf::GeometryPosition{0, 0, 8}};
  const std::array<double, 3> nucleus{0, 0, 0};

  CHECK_NOTHROW(
      pf::PsiFormerGeometryCache(2, pf::GeometryPositionView::interleaved(nucleus.data(), 1), periodic));

  periodic.periodic_axes[2] = false;
  CHECK_THROWS_AS(pf::PsiFormerGeometryCache(
                      2, pf::GeometryPositionView::interleaved(nucleus.data(), 1), periodic),
                  std::invalid_argument);
  periodic.periodic_axes[2] = true;
  periodic.lattice_vectors[2] = periodic.lattice_vectors[1];
  CHECK_THROWS_AS(pf::PsiFormerGeometryCache(
                      2, pf::GeometryPositionView::interleaved(nucleus.data(), 1), periodic),
                  std::invalid_argument);
}

TEST_CASE("PsiFormer two-cell geometry retains simulation pair information",
          "[wavefunction][psiformer][periodic]")
{
  pf::PeriodicGeometryMapping mapping;
  mapping.primitive = {pf::GeometryBoundaryKind::PERIODIC,
                       {{{3.2, 0.4, -0.2}, {0.3, 2.8, 0.5}, {-0.1, 0.6, 3.6}}},
                       {{true, true, true}}};
  mapping.simulation = mapping.primitive;
  mapping.tiling[0][0] = 2;
  for (std::size_t c = 0; c < 3; ++c)
    mapping.simulation.lattice_vectors[0][c] *= 2;
  const std::array<double, 3> motif{0.12, -0.23, 0.31};
  const std::array<double, 6> positions{0.71, 0.52, -0.41, -0.27, 1.18, 0.84};
  pf::PsiFormerGeometryCache cache(2, pf::GeometryPositionView::interleaved(motif.data(), 1), mapping);
  cache.update(pf::GeometryPositionView::interleaved(positions.data(), 2));
  const auto ei = cache.electronNucleusPairs().distances();
  const auto ei_q = cache.electronNucleusPairs().displacements();
  const auto ei_p = cache.electronNucleusPairs().complementaryDisplacements();
  const double ee = cache.electronElectronPairs().distances()[0];
  const auto fingerprint = cache.storageFingerprint();
  const auto bytes = cache.storageBytes();
  auto shifted = cache.electrons()[0];
  for (std::size_t c = 0; c < 3; ++c)
    shifted[c] += mapping.primitive.lattice_vectors[0][c];
  cache.updateElectron(0, shifted);
  for (std::size_t i = 0; i < 2; ++i)
  {
    checkClose(cache.electronNucleusPairs().distances()[i], ei[i], 1e-12);
    for (std::size_t c = 0; c < 3; ++c)
    {
      checkClose(cache.electronNucleusPairs().displacements()[i][c], ei_q[i][c], 1e-12);
      checkClose(cache.electronNucleusPairs().complementaryDisplacements()[i][c], ei_p[i][c], 1e-12);
    }
  }
  CHECK(std::abs(cache.electronElectronPairs().distances()[0] - ee) > 0.1);
  CHECK(cache.storageFingerprint() == fingerprint);
  CHECK(cache.storageBytes() == bytes);
  CHECK(cache.boundary().lattice_vectors == mapping.simulation.lattice_vectors);
  CHECK(cache.electronIonBoundary().lattice_vectors == mapping.primitive.lattice_vectors);

  // A second primitive shift is a full simulation translation for both channels.
  for (std::size_t c = 0; c < 3; ++c)
    shifted[c] += mapping.primitive.lattice_vectors[0][c];
  cache.updateElectron(0, shifted);
  checkClose(cache.electronElectronPairs().distances()[0], ee, 1e-12);
}

TEST_CASE("PsiFormer two-cell skew geometry jets use their respective cells",
          "[wavefunction][psiformer][periodic]")
{
  pf::PeriodicGeometryMapping mapping;
  mapping.primitive = {pf::GeometryBoundaryKind::PERIODIC,
                       {{{3.2, 0.4, -0.2}, {0.3, 2.8, 0.5}, {-0.1, 0.6, 3.6}}},
                       {{true, true, true}}};
  mapping.simulation = mapping.primitive;
  mapping.tiling[1][0] = 1;
  mapping.tiling[1][1] = 2;
  for (std::size_t c = 0; c < 3; ++c)
    mapping.simulation.lattice_vectors[1][c] =
        mapping.primitive.lattice_vectors[0][c] + 2 * mapping.primitive.lattice_vectors[1][c];
  const std::array<double, 3> motif{0.12, -0.23, 0.31};
  const std::array<double, 6> positions{0.71, 0.52, -0.41, -0.27, 1.18, 0.84};
  pf::PsiFormerGeometryCache cache(2, pf::GeometryPositionView::interleaved(motif.data(), 1), mapping);
  cache.update(pf::GeometryPositionView::interleaved(positions.data(), 2));
  const std::array<double, 2> radius{cache.electronNucleusPairs().distances()[0],
                                      cache.electronElectronPairs().distances()[0]};
  const std::array<pf::GeometryPosition, 2> gradient{cache.electronNucleusPairs().distanceGradients()[0],
                                                    cache.electronElectronPairs().distanceGradients()[0]};
  const std::array<double, 2> laplacian{cache.electronNucleusPairs().distanceLaplacians()[0],
                                         cache.electronElectronPairs().distanceLaplacians()[0]};
  namespace pg = qmcplusplus::psiformer::periodic_geometry;
  const auto cells = pg::makeTwoCell(mapping.simulation.lattice_vectors,
      mapping.primitive.lattice_vectors, mapping.simulation.periodic_axes);
  CHECK_NOTHROW(pg::validateCell(cells));
  for (std::size_t channel = 0; channel < 2; ++channel)
  {
    const auto& table = channel == 0 ? cache.electronNucleusPairs() : cache.electronElectronPairs();
    pg::Pair pair;
    REQUIRE(pg::evaluatePair(cells, positions.data(), channel == 0 ? motif.data() : positions.data() + 3,
                             pair, channel == 0));
    checkClose(pair.radius, radius[channel]);
    checkClose(pair.radius_laplacian, laplacian[channel]);
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      checkClose(pair.radius_gradient[dimension], gradient[channel][dimension]);
      checkClose(pair.displacement[dimension], table.displacements()[0][dimension]);
      checkClose(pair.complementary[dimension], table.complementaryDisplacements()[0][dimension]);
      checkClose(pair.laplacian[dimension], table.displacementLaplacians()[0][dimension]);
      checkClose(pair.complementary_laplacian[dimension], table.complementaryDisplacementLaplacians()[0][dimension]);
      for (std::size_t derivative = 0; derivative < 3; ++derivative)
      {
        checkClose(pair.jacobian[dimension][derivative], table.displacementJacobians()[0][dimension][derivative]);
        checkClose(pair.complementary_jacobian[dimension][derivative],
                   table.complementaryDisplacementJacobians()[0][dimension][derivative]);
      }
    }
  }
  auto invalid_cells = cells;
  invalid_cells.ion_reciprocal[0][0] += 0.1;
  CHECK_THROWS_AS(pg::validateCell(invalid_cells), std::invalid_argument);
  for (int orientation : {1, -1})
  {
    pg::LogComplementPairJet actual;
    REQUIRE(pg::evaluateLogComplementPair(cells, positions.data() + (orientation == 1 ? 0 : 3),
        positions.data() + (orientation == 1 ? 3 : 0), -1, actual));
    const auto expected = qmcplusplus::psiformer::periodic_feature::evaluate(cache.electronElectronPairs(),
        0, mapping.simulation, orientation, qmcplusplus::psiformer::PeriodicFeatureTransform::LOG_COMPLEMENT_V1, -1);
    CHECK(actual.differentiable == expected.differentiable);
    for (std::size_t feature = 0; feature < 8; ++feature)
    {
      checkClose(actual.value[feature], expected.value[feature]);
      checkClose(actual.laplacian[feature], expected.laplacian[feature]);
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        checkClose(actual.gradient[feature][dimension], expected.gradient[feature][dimension]);
    }
  }
  for (double step : {2e-4, 1e-4})
  {
    std::array<double, 2> trace{};
    for (std::size_t c = 0; c < 3; ++c)
    {
      auto plus = positions;
      auto minus = positions;
      plus[c] += step;
      minus[c] -= step;
      cache.update(pf::GeometryPositionView::interleaved(plus.data(), 2));
      const std::array<double, 2> rp{cache.electronNucleusPairs().distances()[0],
                                     cache.electronElectronPairs().distances()[0]};
      cache.update(pf::GeometryPositionView::interleaved(minus.data(), 2));
      const std::array<double, 2> rm{cache.electronNucleusPairs().distances()[0],
                                     cache.electronElectronPairs().distances()[0]};
      for (std::size_t channel = 0; channel < 2; ++channel)
      {
        checkClose((rp[channel] - rm[channel]) / (2 * step), gradient[channel][c], 2e-7);
        trace[channel] += (rp[channel] - 2 * radius[channel] + rm[channel]) / (step * step);
      }
    }
    for (std::size_t channel = 0; channel < 2; ++channel)
      checkClose(trace[channel], laplacian[channel], 2e-5);
  }
}

TEST_CASE("PsiFormer two-cell mapping validates and preserves identity geometry",
          "[wavefunction][psiformer][periodic]")
{
  pf::PeriodicGeometryMapping mapping;
  mapping.primitive = {pf::GeometryBoundaryKind::PERIODIC,
                       {{{-1.718, 1.718, 1.718}, {1.718, -1.718, 1.718}, {1.718, 1.718, -1.718}}},
                       {{true, true, true}}};
  mapping.simulation = {pf::GeometryBoundaryKind::PERIODIC,
                        {{{3.436, 0, 0}, {0, 3.436, 0}, {0, 0, 3.436}}},
                        {{true, true, true}}};
  mapping.tiling = {{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}}};
  CHECK_NOTHROW(pf::validatePeriodicGeometryMapping(mapping));
  const std::array<double, 3> one_site{0, 0, 0};
  const std::array<double, 6> conventional_sites{0, 0, 0, 1.718, 1.718, 1.718};
  const auto one_view = pf::GeometryPositionView::interleaved(one_site.data(), 1);
  const auto conventional_view = pf::GeometryPositionView::interleaved(conventional_sites.data(), 2);
  CHECK(pf::mapPeriodicIonMotif(mapping, one_view, {3}, conventional_view, {3, 3}) ==
        std::vector<std::size_t>{0, 0});
  CHECK_THROWS_AS(pf::mapPeriodicIonMotif(mapping, one_view, {3}, conventional_view, {3, 1}),
                  std::invalid_argument);
  CHECK_THROWS_AS(pf::mapPeriodicIonMotif(mapping, one_view, {3}, one_view, {3}),
                  std::invalid_argument);
  auto duplicate_sites = conventional_sites;
  for (std::size_t c = 0; c < 3; ++c)
    duplicate_sites[3 + c] = mapping.simulation.lattice_vectors[0][c];
  CHECK_THROWS_AS(pf::mapPeriodicIonMotif(mapping, one_view, {3},
      pf::GeometryPositionView::interleaved(duplicate_sites.data(), 2), {3, 3}), std::invalid_argument);
  auto displaced_sites = conventional_sites;
  displaced_sites[3] += 0.1;
  CHECK_THROWS_AS(pf::mapPeriodicIonMotif(mapping, one_view, {3},
      pf::GeometryPositionView::interleaved(displaced_sites.data(), 2), {3, 3}), std::invalid_argument);
  auto wrapped_sites = conventional_sites;
  for (std::size_t c = 0; c < 3; ++c)
    wrapped_sites[3 + c] -= mapping.simulation.lattice_vectors[1][c];
  CHECK(pf::mapPeriodicIonMotif(mapping, one_view, {3},
      pf::GeometryPositionView::interleaved(wrapped_sites.data(), 2), {3, 3}) == std::vector<std::size_t>{0, 0});
  auto invalid = mapping;
  invalid.tiling[0][0] = 1;
  CHECK_THROWS_AS(pf::validatePeriodicGeometryMapping(invalid), std::invalid_argument);
  invalid = mapping;
  invalid.primitive.periodic_axes[0] = false;
  CHECK_THROWS_AS(pf::validatePeriodicGeometryMapping(invalid), std::invalid_argument);
  invalid = mapping;
  invalid.simulation.lattice_vectors[2] = invalid.simulation.lattice_vectors[1];
  CHECK_THROWS_AS(pf::validatePeriodicGeometryMapping(invalid), std::invalid_argument);

  mapping.primitive = mapping.simulation;
  mapping.tiling = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
  const std::array<double, 6> motif{0, 0, 0, 1.718, 1.718, 1.718};
  const std::array<double, 6> electrons{0.4, 0.6, 0.9, -0.3, 1.2, 0.2};
  pf::PsiFormerGeometryCache legacy(2, pf::GeometryPositionView::interleaved(motif.data(), 2), mapping.simulation);
  pf::PsiFormerGeometryCache selected(2, pf::GeometryPositionView::interleaved(motif.data(), 2), mapping);
  legacy.update(pf::GeometryPositionView::interleaved(electrons.data(), 2));
  selected.update(pf::GeometryPositionView::interleaved(electrons.data(), 2));
  CHECK(legacy.electronNucleusPairs().distances() == selected.electronNucleusPairs().distances());
  CHECK(legacy.electronNucleusPairs().distanceGradients() == selected.electronNucleusPairs().distanceGradients());
  CHECK(legacy.electronNucleusPairs().distanceLaplacians() == selected.electronNucleusPairs().distanceLaplacians());
  CHECK(legacy.electronElectronPairs().distances() == selected.electronElectronPairs().distances());
  CHECK(legacy.storageBytes() == selected.storageBytes());
}

TEST_CASE("Periodic feature policies and ordered pair jets match independent skew formulas",
          "[wavefunction][psiformer][periodic_profile]")
{
  using qmcplusplus::psiformer::PeriodicFeatureTransform;
  namespace feature = qmcplusplus::psiformer::periodic_feature;
  const pf::GeometryBoundary cell{pf::GeometryBoundaryKind::PERIODIC,
      {{{3.1, 0, 0}, {0.7, 2.9, 0}, {-0.2, 0.4, 3.6}}}, {{true, true, true}}};
  const std::array<double, 3> origin{};
  const pf::GeometryPosition x{0.71, -0.43, 0.61};
  pf::PsiFormerGeometryCache cache(1, pf::GeometryPositionView::interleaved(origin.data(), 1), cell);
  cache.update(pf::GeometryPositionView::interleaved(x.data(), 1));
  // Solve this triangular lattice explicitly; the oracle does not use cache vectors or reciprocal helpers.
  const auto oracle = [&](const pf::GeometryPosition& displacement, PeriodicFeatureTransform transform) {
    constexpr double pi2 = 6.283185307179586476925286766559;
    pf::GeometryPosition fractional;
    fractional[2] = displacement[2] / 3.6;
    fractional[1] = (displacement[1] - 0.4 * fractional[2]) / 2.9;
    fractional[0] = (displacement[0] - 0.7 * fractional[1] + 0.2 * fractional[2]) / 3.1;
    pf::GeometryPosition sine{}, complement{}, cosine{};
    for (std::size_t a = 0; a < 3; ++a)
      for (std::size_t c = 0; c < 3; ++c)
      {
        const double dual = cell.lattice_vectors[a][c] / pi2;
        sine[c] += dual * std::sin(pi2 * fractional[a]);
        complement[c] += dual * (1 - std::cos(pi2 * fractional[a]));
        cosine[c] += dual * std::cos(pi2 * fractional[a]);
      }
    double radius2 = 0;
    for (std::size_t c = 0; c < 3; ++c)
      radius2 += sine[c] * sine[c] + complement[c] * complement[c];
    const double radius = std::sqrt(radius2);
    const bool raw = transform == PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1;
    std::array<double, 8> result{};
    result[0] = raw ? radius : std::log1p(radius);
    result[7] = -1;
    const double scale = raw ? 1 : std::log1p(radius) / radius;
    for (std::size_t c = 0; c < 3; ++c)
    {
      result[1 + c] = scale * sine[c];
      result[4 + c] = raw ? cosine[c] : scale * complement[c];
    }
    return result;
  };
  for (auto policy : {PeriodicFeatureTransform::RAW_TRIGONOMETRIC_V1, PeriodicFeatureTransform::LOG_COMPLEMENT_V1})
    for (int orientation : {1, -1})
    {
      const auto jet = feature::evaluate(cache.electronNucleusPairs(), 0, cell, orientation, policy, -1);
      REQUIRE(jet.differentiable);
      auto point = x;
      for (double& c : point)
        c *= orientation;
      const auto value = oracle(point, policy);
      for (std::size_t f = 0; f < value.size(); ++f)
        checkClose(jet.value[f], value[f], 1e-12);
      for (double step : {2e-4, 1e-4})
      {
        std::array<double, 8> trace{};
        for (std::size_t d = 0; d < 3; ++d)
        {
          auto plus = point, minus = point;
          plus[d] += step;
          minus[d] -= step;
          const auto vp = oracle(plus, policy), vm = oracle(minus, policy);
          for (std::size_t f = 0; f < value.size(); ++f)
          {
            checkClose((vp[f] - vm[f]) / (2 * step), jet.gradient[f][d], 2e-7);
            trace[f] += (vp[f] - 2 * value[f] + vm[f]) / (step * step);
          }
        }
        for (std::size_t f = 0; f < value.size(); ++f)
          checkClose(trace[f], jet.laplacian[f], 2e-5);
      }
    }
  cache.update(pf::GeometryPositionView::interleaved(origin.data(), 1));
  const auto coalescent = feature::evaluate(cache.electronNucleusPairs(), 0, cell, 1,
      PeriodicFeatureTransform::LOG_COMPLEMENT_V1, 1);
  CHECK_FALSE(coalescent.differentiable);
  CHECK(coalescent.value[0] == 0);
  CHECK(coalescent.value[7] == 1);
}
