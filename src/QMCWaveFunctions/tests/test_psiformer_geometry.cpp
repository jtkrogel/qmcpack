//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "QMCWaveFunctions/PsiFormer/PsiFormerGeometry.h"

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

TEST_CASE("PsiFormer rejects unavailable periodic geometry", "[wavefunction][psiformer]")
{
  CHECK(pf::supportsGeometryBoundary(pf::GeometryBoundaryKind::OPEN));
  CHECK_FALSE(pf::supportsGeometryBoundary(pf::GeometryBoundaryKind::PERIODIC));

  pf::GeometryBoundary periodic;
  periodic.kind          = pf::GeometryBoundaryKind::PERIODIC;
  periodic.periodic_axes = {true, true, true};
  periodic.lattice_vectors =
      {pf::GeometryPosition{8, 0, 0}, pf::GeometryPosition{0, 8, 0}, pf::GeometryPosition{0, 0, 8}};
  const std::array<double, 3> nucleus{0, 0, 0};

  CHECK_THROWS_AS(
      pf::PsiFormerGeometryCache(2, pf::GeometryPositionView::interleaved(nucleus.data(), 1), periodic),
      std::invalid_argument);
}
