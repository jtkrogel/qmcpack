//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerFeatureCoordinateReverse.h"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace p = qmcplusplus::psiformer;

TEST_CASE("PsiFormer raw feature coordinate transpose matches forward jets and differences",
          "[psiformer][coordinate_reverse]")
{
  bool periodic = false;
  SECTION("open") {}
  SECTION("skew periodic seven-channel policy") { periodic = true; }
  const auto cell = p::periodic_geometry::makeCell(
      {{{3.1, 0.2, -0.1}, {0.4, 2.8, 0.3}, {-0.2, 0.5, 3.4}}}, {true, true, true});
  const auto* boundary = periodic ? &cell : nullptr;
  const auto make_layout = [&](p::SpatialJetMode mode) {
    return periodic ? p::makePeriodicFeatureJetLayout(2, 2, 2, 1, mode, 4, 10, 4)
                    : p::makeOpenFeatureJetLayout(2, 2, 2, 1, mode, 4, 10, 4);
  };
  const auto value = make_layout(p::SpatialJetMode::VALUE_ONLY);
  const auto full = make_layout(p::SpatialJetMode::FULL_VGL);
  std::vector<double> positions{0.4, 0.2, 0.3, 99, -0.2, 0.6, 0.1, 99, 99, 99,
                                0.7, -0.3, 0.2, 99, 0.1, 0.8, -0.4, 99, 99, 99};
  const double nuclei[]{0, 0, 0, 99, 1.2, 0.1, -0.2, 99};
  std::vector<double> bars(p::spatialJetSpanElements(value.output));
  for (std::size_t i = 0; i < bars.size(); ++i)
    bars[i] = 0.2 * std::sin(0.4 + 0.13 * i);
  std::vector<double> jets(p::spatialJetSpanElements(full.output));
  std::vector<double> coordinate(12), direction(12);
  for (std::size_t i = 0; i < direction.size(); ++i)
    direction[i] = 0.1 * std::cos(0.2 + 0.7 * i);

  // The independently retained forward Jacobian supplies every coordinate row;
  // the new transpose consumes only a single local raw-feature cotangent row.
  for (std::size_t configuration = 0; configuration < 2; ++configuration)
  {
    REQUIRE(p::open_spatial::buildOpenFeatureJetsForConfiguration(full, positions.data(), nuclei,
        configuration, 0, jets.data(), boundary) == p::OpenSpatialStatus::REGULAR);
    for (std::size_t electron = 0; electron < 2; ++electron)
    {
      auto* result = coordinate.data() + 3 * (2 * configuration + electron);
      REQUIRE(p::feature_coordinate::coordinateAdjointForElectron(value, positions.data(), nuclei,
          bars.data(), configuration, electron, result, boundary) == p::OpenSpatialStatus::REGULAR);
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        double expected = 0;
        for (std::size_t feature = 0; feature < value.output.element_count; ++feature)
          expected += bars[value.output.uncheckedValueOffset(configuration, feature)] *
              jets[full.output.uncheckedGradientOffset(configuration, 3 * electron + dimension, feature)];
        CHECK(result[dimension] == Approx(expected).margin(3e-14));
      }

      // Spin is a discrete input label: changing only its cotangent contributes
      // exactly zero Cartesian gradient for either spin channel.
      const auto spin = value.output.uncheckedValueOffset(configuration,
          (electron + 1) * value.feature_width - 1);
      std::vector<double> spin_only(bars.size(), 0.0);
      spin_only[spin] = 7;
      double spin_shifted[3]{};
      REQUIRE(p::feature_coordinate::coordinateAdjointForElectron(value, positions.data(), nuclei,
          spin_only.data(), configuration, electron, spin_shifted, boundary) == p::OpenSpatialStatus::REGULAR);
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
        CHECK(spin_shifted[dimension] == 0.0);
    }
  }

  const auto objective = [&](const std::vector<double>& shifted) {
    std::vector<double> features(p::spatialJetSpanElements(value.output));
    for (std::size_t configuration = 0; configuration < 2; ++configuration)
      REQUIRE(p::open_spatial::buildOpenFeatureJetsForConfiguration(value, shifted.data(), nuclei,
          configuration, 0, features.data(), boundary) == p::OpenSpatialStatus::REGULAR);
    double result = 0;
    for (std::size_t i = 0; i < features.size(); ++i)
      result += features[i] * bars[i];
    return result;
  };
  auto minus = positions, plus = positions;
  constexpr double delta = 1e-6;
  double transpose_product = 0;
  for (std::size_t configuration = 0; configuration < 2; ++configuration)
    for (std::size_t electron = 0; electron < 2; ++electron)
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const auto flat = 3 * (2 * configuration + electron) + dimension;
        const auto position = configuration * 10 + electron * 4 + dimension;
        minus[position] -= delta * direction[flat];
        plus[position] += delta * direction[flat];
        transpose_product += coordinate[flat] * direction[flat];
      }
  CHECK(transpose_product == Approx((objective(plus) - objective(minus)) / (2 * delta)).margin(3e-8));

  // Failed leaves publish no partial coordinates, including coalescence in a
  // value-only raw layout: this operation always requests first derivatives.
  double guarded[]{913, 913, 913};
  auto coalesced = positions;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    coalesced[dimension] = nuclei[dimension];
  CHECK(p::feature_coordinate::coordinateAdjointForElectron(value, coalesced.data(), nuclei,
      bars.data(), 0, 0, guarded, boundary) == p::OpenSpatialStatus::COALESCENCE);
  bars[value.feature_width - 1] = std::numeric_limits<double>::quiet_NaN();
  CHECK(p::feature_coordinate::coordinateAdjointForElectron(value, positions.data(), nuclei,
      bars.data(), 0, 0, guarded, boundary) == p::OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT);
  for (const double element : guarded)
    CHECK(element == 913);
}

// Keep the geometry oracle independent of the application wavefunction library.
int main(int argc, char* argv[])
{
  return Catch::Session().run(argc, argv);
}
