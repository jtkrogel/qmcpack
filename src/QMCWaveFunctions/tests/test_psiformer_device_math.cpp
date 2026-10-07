//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_device_math.cpp
 * @brief CPU independent-oracle tests for host-callable CUDA/HIP mathematical cores.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_session.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceMath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace math = qmcplusplus::psiformer::device_math;

namespace
{

void checkClose(double actual, long double expected, double tolerance = 3.0e-13)
{
  CHECK(actual == Catch::Approx(static_cast<double>(expected)).epsilon(tolerance).margin(tolerance));
}

long double softenedValueOracle(long double radius)
{
  return radius == 0 ? 1.0L : std::log1p(radius) / radius;
}

long double finiteDifferenceFirst(long double radius)
{
  const long double step = 2.0e-5L * std::max(1.0L, radius);
  return (softenedValueOracle(radius + step) - softenedValueOracle(radius - step)) / (2 * step);
}

long double finiteDifferenceSecond(long double radius)
{
  const long double step = 2.0e-4L * std::max(1.0L, radius);
  return (softenedValueOracle(radius + step) - 2 * softenedValueOracle(radius) +
          softenedValueOracle(radius - step)) /
      (step * step);
}

} // namespace

TEST_CASE("PsiFormer device radial core matches independent scalar formulas", "[psiformer][device][math]")
{
  for (double radius : {0.0, 1.0e-12, 5.0e-4, 0.009, 0.01, 0.4, 3.0, 40.0})
  {
    const auto factors = math::softenedRadial(radius);
    checkClose(factors.log1p_radius, std::log1p(static_cast<long double>(radius)));
    checkClose(factors.log1p_over_radius, softenedValueOracle(radius));
    checkClose(factors.log1p_first, 1.0L / (1.0L + radius));
    checkClose(factors.log1p_second, -1.0L / ((1.0L + radius) * (1.0L + radius)));
  }

  const auto at_origin = math::softenedRadial(0.0);
  checkClose(at_origin.log1p_over_radius_first, -0.5L);
  checkClose(at_origin.log1p_over_radius_second, 2.0L / 3.0L);

  for (double radius : {0.05, 0.4, 2.0})
  {
    const auto factors = math::softenedRadial(radius);
    checkClose(factors.log1p_over_radius_first, finiteDifferenceFirst(radius), 2.0e-8);
    checkClose(factors.log1p_over_radius_second, finiteDifferenceSecond(radius), 2.0e-7);
  }
}

TEST_CASE("PsiFormer open and periodic pair features use the CPU layout contract",
          "[psiformer][device][math]")
{
  const std::array<double, 3> displacement{0.25, -0.5, 0.75};
  const std::array<double, 3> complementary{0.1, 0.2, -0.3};
  const double open_radius = std::sqrt(0.25 * 0.25 + 0.5 * 0.5 + 0.75 * 0.75);
  const double periodic_radius = std::sqrt(open_radius * open_radius + 0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3);

  std::array<double, 7> features{};
  math::assemblePairFeatures(displacement.data(), static_cast<const double*>(nullptr),
                             open_radius, false, features.data());
  const long double open_factor = softenedValueOracle(open_radius);
  checkClose(features[0], std::log1p(static_cast<long double>(open_radius)));
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    checkClose(features[1 + dimension], displacement[dimension] * open_factor);

  math::assemblePairFeatures(displacement.data(), complementary.data(), periodic_radius, true,
                             features.data());
  const long double periodic_factor = softenedValueOracle(periodic_radius);
  checkClose(features[0], std::log1p(static_cast<long double>(periodic_radius)));
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
  {
    checkClose(features[1 + dimension], displacement[dimension] * periodic_factor);
    checkClose(features[4 + dimension], complementary[dimension] * periodic_factor);
  }

  const std::array<double, 3> zero{};
  math::assemblePairFeatures(zero.data(), static_cast<const double*>(nullptr),
                             0.0, false, features.data());
  for (std::size_t element = 0; element < 4; ++element)
    checkClose(features[element], 0.0L);
}

TEST_CASE("PsiFormer elementwise jets match finite differences", "[psiformer][device][math]")
{
  for (const math::ScalarJet<double> input :
       {math::ScalarJet<double>{-3.0, 0.2, -0.4}, {0.0, -0.8, 0.3}, {2.5, 1.1, -0.7}})
  {
    const auto result = math::tanhJet(input);
    const long double value = std::tanh(static_cast<long double>(input.value));
    const long double first_factor = 1 - value * value;
    checkClose(result.value, value);
    checkClose(result.first, first_factor * input.first);
    checkClose(result.second,
               first_factor * input.second - 2 * value * first_factor * input.first * input.first);
  }

  const auto sum = math::addJets(math::ScalarJet<double>{1.0, -2.0, 3.0},
                                 math::ScalarJet<double>{-4.0, 5.0, -6.0});
  CHECK(sum.value == -3.0);
  CHECK(sum.first == 3.0);
  CHECK(sum.second == -3.0);
}

TEST_CASE("PsiFormer envelope and cusp cores match direct CPU oracles", "[psiformer][device][math]")
{
  for (const auto& values : std::vector<std::array<double, 3>>{{0.0, 2.0, -1.5},
                                                               {0.7, -0.25, 3.0},
                                                               {8.0, 1.25, -0.2}})
  {
    const long double expected = static_cast<long double>(values[1]) *
        std::exp(-std::abs(static_cast<long double>(values[2]) * values[0]));
    checkClose(math::envelopeContribution(values[0], values[1], values[2]), expected);
  }

  for (const auto& values : std::vector<std::array<double, 3>>{{0.0, 1.2, 0.25},
                                                               {0.6, 2.0, 0.5},
                                                               {20.0, 0.3, 0.25}})
  {
    const long double expected = -static_cast<long double>(values[2]) * values[1] * values[1] /
        (values[1] + values[0]);
    checkClose(math::cuspPair(values[0], values[1], values[2]), expected);
  }
}

// Keep this oracle executable independent of QMCPACK's accelerator-initializing
// Catch main so it remains runnable in CUDA/HIP builds on driverless hosts.
int main(int argc, char* argv[])
{
  return Catch::Session().run(argc, argv);
}
