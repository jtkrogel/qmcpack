//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include "QMCWaveFunctions/PsiFormer/PsiFormerOpenOrbital.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace qmcplusplus::psiformer
{
namespace
{
struct PairFixture
{
  OpenOrbitalJetLayout layout;
  periodic_geometry::Cell cell;
  std::vector<double> positions, nuclei, features, weight, pi, zeta, projected, radius;
  std::vector<std::uint8_t> valid;
  PairFixture(std::size_t n, std::size_t up, std::size_t determinants)
      : layout(makeOpenOrbitalJetLayout(3, determinants, n, 4, 2, up, SpatialJetMode::VALUE_ONLY,
                                       3, n + 1, 0, 0, 0, 0, 0, 4, 0, 4)),
        cell(periodic_geometry::makeTwoCell({{{12., .4, .2}, {.2, 14., .6}, {.4, .2, 16.}}},
                                           {{{6., .2, .1}, {.1, 7., .3}, {.2, .1, 8.}}},
                                           {true, true, true})),
        positions(3 * layout.position_configuration_stride), nuclei(4 * layout.nucleus_row_stride),
        features(spatialJetSpanElements(layout.features), .17), weight(openOrbitalBackflowSpan(layout), .13),
        pi(openOrbitalEnvelopeSpan(layout), .3), zeta(pi.size(), -.7),
        projected(spatialJetSpanElements(layout.orbitals), .23), radius(3 * n * 4), valid(radius.size())
  {
    for (std::size_t b = 0; b < 3; ++b)
      for (std::size_t e = 0; e < n; ++e)
        for (std::size_t d = 0; d < 3; ++d)
          positions[b * layout.position_configuration_stride + e * layout.position_row_stride + d] =
              .1 + .021 * e + .13 * d + .03 * b;
    for (std::size_t i = 0; i < 4; ++i)
      for (std::size_t d = 0; d < 3; ++d) nuclei[i * layout.nucleus_row_stride + d] = .4 + .17 * i + .11 * d;
  }
  PeriodicValuePairView produce()
  {
    const auto n = layout.features.electron_count;
    for (std::size_t b = 0; b < 3; ++b)
      for (std::size_t e = 0; e < n; ++e)
        for (std::size_t i = 0; i < 4; ++i)
        {
          const auto index = (b * n + e) * 4 + i;
          periodic_geometry::Pair pair;
          valid[index] = periodic_geometry::evaluatePair(cell,
              positions.data() + b * layout.position_configuration_stride + e * layout.position_row_stride,
              nuclei.data() + i * layout.nucleus_row_stride, pair, true);
          radius[index] = valid[index] ? pair.radius : 0.;
        }
    return {radius.data(), valid.data(), radius.size()};
  }
  OpenSpatialStatus element(std::size_t b, std::size_t det, std::size_t row, std::size_t col,
                            double* output, bool envelope_only, PeriodicValuePairView bank)
  {
    return open_orbital::buildElement(layout, features.data(), positions.data(), nuclei.data(),
        weight.data(), weight.data(), pi.data(), pi.data(), zeta.data(), zeta.data(), b, 0, det, row, col,
        output, nullptr, nullptr, &cell, envelope_only ? nullptr : projected.data(), envelope_only,
        nullptr, nullptr, nullptr, bank);
  }
  void compare()
  {
    const auto bank = produce();
    const auto count = spatialJetSpanElements(layout.orbitals), n = layout.features.electron_count;
    std::vector<double> baseline(count, -71.), candidate(count, -71.);
    for (const bool envelope_only : {false, true})
    {
      for (std::size_t b = 0; b < 3; ++b)
        for (std::size_t det = 0; det < layout.determinant_count; ++det)
          for (std::size_t row = 0; row < n; ++row)
            for (std::size_t col = 0; col < n; ++col)
              REQUIRE(element(b, det, row, col, baseline.data(), envelope_only, {}) ==
                      element(b, det, row, col, candidate.data(), envelope_only, bank));
      REQUIRE(std::memcmp(baseline.data(), candidate.data(), count * sizeof(double)) == 0);
    }
  }
};
}

TEST_CASE("Periodic VALUE Pair bank shares the canonical envelope leaf", "[psiformer][periodic_value_pairs]")
{
  for (const auto determinants : {16u, 64u}) PairFixture(44, 26, determinants).compare();
  PairFixture(64, 32, 2).compare();
  PairFixture(3, 0, 2).compare();
  PairFixture(3, 3, 2).compare();
}

TEST_CASE("Periodic VALUE Pair bank preserves physical rejection and coalescence", "[psiformer][periodic_value_pairs]")
{
  PairFixture f(3, 1, 2);
  const auto original = f.positions;
  // Exact and near coalescence, signed zero, primitive-cell translation and tails.
  std::copy_n(f.nuclei.data(), 3, f.positions.data());
  f.compare();
  f.positions[0] += 1e-13;
  f.compare();
  f.positions[0] = -0.;
  f.compare();
  for (std::size_t d = 0; d < 3; ++d) f.positions[d] += f.cell.ion_lattice[1][d];
  f.compare();
  for (const auto invalid : {std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN(),
                             std::numeric_limits<double>::max()})
  {
    f.positions = original;
    f.positions[0] = invalid;
    f.nuclei[0] = -std::numeric_limits<double>::max();
    f.compare();
  }
  f.positions = original;
  f.nuclei[0] = .4;
  f.pi[0] = std::numeric_limits<double>::quiet_NaN();
  f.compare();
  f.pi[0] = .3;
  f.compare(); // Failed producer/consumer data must be overwritten on recovery.
  const auto bank = f.produce();
  std::vector<double> out(spatialJetSpanElements(f.layout.orbitals), -19.);
  for (const auto malformed : {PeriodicValuePairView{bank.radii, nullptr, bank.pair_count},
                               PeriodicValuePairView{nullptr, bank.valid, bank.pair_count},
                               PeriodicValuePairView{bank.radii, bank.valid, 0}})
    REQUIRE(f.element(0, 0, 0, 0, out.data(), false, malformed) == OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT);
  REQUIRE(out[0] == 0.);
  f.layout.features.mode = SpatialJetMode::FULL_VGL;
  REQUIRE(f.element(0, 0, 0, 0, out.data(), false, bank) == OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT);
  REQUIRE(out[0] == 0.);
}
} // namespace qmcplusplus::psiformer
int main(int argc, char* argv[]) { return Catch::Session().run(argc, argv); }
