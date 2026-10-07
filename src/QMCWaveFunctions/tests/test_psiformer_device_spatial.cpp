//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_device_spatial.cpp
 * @brief Independent CPU oracle for the portable PsiFormer spatial-jet foundation.
 */

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialLayout.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace psiformer = qmcplusplus::psiformer;
namespace math = qmcplusplus::psiformer::device_math;

namespace
{

void checkClose(double actual, long double expected, double tolerance = 2.0e-12)
{
  CHECK(actual == Catch::Approx(static_cast<double>(expected))
                      .epsilon(tolerance).margin(tolerance));
}

std::vector<long double> referenceSoftmax(const std::vector<long double>& logits)
{
  const long double maximum = *std::max_element(logits.begin(), logits.end());
  std::vector<long double> probabilities(logits.size());
  long double normalization = 0.0L;
  for (std::size_t column = 0; column < logits.size(); ++column)
  {
    probabilities[column] = std::exp(logits[column] - maximum);
    normalization += probabilities[column];
  }
  for (long double& probability : probabilities)
    probability /= normalization;
  return probabilities;
}

} // namespace

TEST_CASE("PsiFormer spatial layouts validate active, full, and padded storage",
          "[psiformer][device][spatial]")
{
  const psiformer::SpatialJetLayout active = psiformer::makeSpatialJetLayout(
      2, 7, 4, psiformer::SpatialJetMode::ACTIVE, 9, 40);
  CHECK(active.gradient_lanes == 3);
  CHECK(active.laplacian_lanes == 0);
  CHECK(active.plane_count == 4);
  CHECK(psiformer::spatialJetSpanElements(active) == 74);
  CHECK(psiformer::checkedSpatialValueOffset(active, 1, 6) == 46);
  CHECK(psiformer::checkedSpatialGradientOffset(active, 1, 2, 6) == 73);

  const psiformer::SpatialJetLayout full = psiformer::makeSpatialJetLayout(
      2, 5, 2, psiformer::SpatialJetMode::FULL_VGL, 8, 75);
  CHECK(full.gradient_lanes == 6);
  CHECK(full.laplacian_lanes == 2);
  CHECK(full.plane_count == 9);
  CHECK(psiformer::spatialJetSpanElements(full) == 144);
  CHECK(psiformer::checkedSpatialGradientOffset(full, 1, 5, 4) == 127);
  CHECK(psiformer::checkedSpatialLaplacianOffset(full, 1, 1, 4) == 143);

  psiformer::SpatialJetLayout inconsistent = full;
  inconsistent.gradient_lanes = 5;
  CHECK_THROWS_AS(psiformer::validateSpatialJetLayout(inconsistent), std::invalid_argument);
  inconsistent = full;
  inconsistent.configuration_stride = 68;
  CHECK_THROWS_AS(psiformer::validateSpatialJetLayout(inconsistent), std::invalid_argument);
  CHECK_THROWS_AS(psiformer::checkedSpatialGradientOffset(full, 0, 6, 0), std::out_of_range);
  CHECK_THROWS_AS(psiformer::checkedSpatialLaplacianOffset(active, 0, 0, 0), std::out_of_range);
  CHECK_THROWS_AS(psiformer::checkedSpatialValueOffset(full, 2, 0), std::out_of_range);
  CHECK_THROWS_AS(psiformer::makeSpatialJetLayout(
                      std::numeric_limits<std::size_t>::max(), 2, 1,
                      psiformer::SpatialJetMode::ACTIVE),
                  std::length_error);
}

TEST_CASE("PsiFormer product jet retains the trace-gradient cross term",
          "[psiformer][device][spatial]")
{
  constexpr double left_value = 1.2;
  constexpr double right_value = -0.8;
  const std::array<double, 3> left_gradient{0.3, -0.2, 0.5};
  const std::array<double, 3> right_gradient{0.4, 0.1, -0.3};
  const std::array<double, 3> left_second{0.2, 0.1, 0.4};
  const std::array<double, 3> right_second{-0.2, -0.1, -0.3};
  constexpr double left_laplacian = 0.7;
  constexpr double right_laplacian = -0.6;
  double output_value = 0.0;
  std::array<double, 3> output_gradient{};
  double output_laplacian = 0.0;
  math::productJet(left_value, left_gradient.data(), left_laplacian,
                   right_value, right_gradient.data(), right_laplacian,
                   3, &output_value, output_gradient.data(), &output_laplacian);

  checkClose(output_value, static_cast<long double>(left_value) * right_value);
  long double gradient_dot = 0.0L;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
  {
    checkClose(output_gradient[dimension],
               static_cast<long double>(left_gradient[dimension]) * right_value +
                   static_cast<long double>(left_value) * right_gradient[dimension]);
    gradient_dot += static_cast<long double>(left_gradient[dimension]) *
        right_gradient[dimension];
  }
  checkClose(output_laplacian,
             static_cast<long double>(left_laplacian) * right_value +
                 2.0L * gradient_dot +
                 static_cast<long double>(left_value) * right_laplacian);

  constexpr long double step = 2.0e-4L;
  long double finite_difference_trace = 0.0L;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
  {
    const long double left_plus = left_value + step * left_gradient[dimension] +
        0.5L * step * step * left_second[dimension];
    const long double left_minus = left_value - step * left_gradient[dimension] +
        0.5L * step * step * left_second[dimension];
    const long double right_plus = right_value + step * right_gradient[dimension] +
        0.5L * step * step * right_second[dimension];
    const long double right_minus = right_value - step * right_gradient[dimension] +
        0.5L * step * step * right_second[dimension];
    finite_difference_trace +=
        (left_plus * right_plus - 2.0L * left_value * right_value +
         left_minus * right_minus) / (step * step);
  }
  checkClose(output_laplacian, finite_difference_trace, 2.0e-8);

  std::array<double, 3> aliased_gradient = left_gradient;
  double aliased_value = left_value;
  double aliased_laplacian = left_laplacian;
  math::productJet(aliased_value, aliased_gradient.data(), aliased_laplacian,
                   right_value, right_gradient.data(), right_laplacian,
                   3, &aliased_value, aliased_gradient.data(), &aliased_laplacian);
  checkClose(aliased_laplacian, output_laplacian);
}

TEST_CASE("PsiFormer softmax jets preserve padded B greater than one rows",
          "[psiformer][device][spatial]")
{
  constexpr std::size_t configurations = 2;
  constexpr std::size_t rows           = 2;
  constexpr std::size_t width          = 3;
  constexpr std::size_t row_stride     = 5;
  constexpr double sentinel            = -91.0;
  const psiformer::SoftmaxJetRowLayout layout = psiformer::makeSoftmaxJetRowLayout(
      configurations, rows, width, 1, psiformer::SpatialJetMode::FULL_VGL,
      row_stride, 13, 67);
  std::vector<double> jets(psiformer::spatialJetSpanElements(layout.jets), sentinel);
  const std::array<std::array<double, width>, configurations * rows> logits{{
      {1000.0, 999.0, -1000.0},
      {-800.0, -800.0, -800.0},
      {5.0, -2.0, 4.5},
      {-1000.0, 1000.0, 999.5}}};

  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t column = 0; column < width; ++column)
      {
        const std::size_t element = row * row_stride + column;
        jets[layout.jets.uncheckedValueOffset(configuration, element)] =
            logits[configuration * rows + row][column];
        for (std::size_t lane = 0; lane < 3; ++lane)
          jets[layout.jets.uncheckedGradientOffset(configuration, lane, element)] =
              0.04 * static_cast<double>(1 + configuration + 2 * row + column) -
              0.03 * static_cast<double>(lane);
        jets[layout.jets.uncheckedLaplacianOffset(configuration, 0, element)] =
            -0.05 + 0.02 * static_cast<double>(configuration + row + column);
      }
  const std::vector<double> input = jets;
  psiformer::stableSoftmaxJetRows(layout, jets.data());

  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
    for (std::size_t row = 0; row < rows; ++row)
    {
      std::vector<long double> row_logits(width);
      for (std::size_t column = 0; column < width; ++column)
        row_logits[column] = logits[configuration * rows + row][column];
      const std::vector<long double> probability = referenceSoftmax(row_logits);
      std::array<long double, 3> mean_gradient{};
      long double mean_laplacian = 0.0L;
      for (std::size_t column = 0; column < width; ++column)
      {
        const std::size_t element = row * row_stride + column;
        mean_laplacian += probability[column] * input[
            layout.jets.uncheckedLaplacianOffset(configuration, 0, element)];
        for (std::size_t lane = 0; lane < 3; ++lane)
          mean_gradient[lane] += probability[column] * input[
              layout.jets.uncheckedGradientOffset(configuration, lane, element)];
      }
      long double mean_squared_deviation = 0.0L;
      for (std::size_t column = 0; column < width; ++column)
      {
        const std::size_t element = row * row_stride + column;
        long double squared_deviation = 0.0L;
        for (std::size_t lane = 0; lane < 3; ++lane)
        {
          const long double deviation = input[
              layout.jets.uncheckedGradientOffset(configuration, lane, element)] -
              mean_gradient[lane];
          squared_deviation += deviation * deviation;
        }
        mean_squared_deviation += probability[column] * squared_deviation;
      }

      for (std::size_t column = 0; column < width; ++column)
      {
        const std::size_t element = row * row_stride + column;
        checkClose(jets[layout.jets.uncheckedValueOffset(configuration, element)],
                   probability[column]);
        long double squared_deviation = 0.0L;
        for (std::size_t lane = 0; lane < 3; ++lane)
        {
          const long double original_gradient = input[
              layout.jets.uncheckedGradientOffset(configuration, lane, element)];
          const long double deviation = original_gradient - mean_gradient[lane];
          squared_deviation += deviation * deviation;
          checkClose(jets[layout.jets.uncheckedGradientOffset(configuration, lane, element)],
                     probability[column] * deviation);
        }
        const long double original_laplacian = input[
            layout.jets.uncheckedLaplacianOffset(configuration, 0, element)];
        checkClose(jets[layout.jets.uncheckedLaplacianOffset(configuration, 0, element)],
                   probability[column] *
                       (original_laplacian - mean_laplacian + squared_deviation -
                        mean_squared_deviation));
      }
    }

  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
    for (std::size_t plane = 0; plane < layout.jets.plane_count; ++plane)
      for (std::size_t row = 0; row + 1 < rows; ++row)
        for (std::size_t padding = width; padding < row_stride; ++padding)
          CHECK(jets[layout.jets.uncheckedPlaneOffset(
                    configuration, plane, row * row_stride + padding)] == sentinel);

  constexpr std::size_t finite_difference_configuration = 1;
  constexpr std::size_t finite_difference_row = 0;
  constexpr long double step = 2.0e-4L;
  std::array<std::vector<long double>, 3> plus_probability;
  std::array<std::vector<long double>, 3> minus_probability;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
  {
    std::vector<long double> plus_logits(width);
    std::vector<long double> minus_logits(width);
    for (std::size_t column = 0; column < width; ++column)
    {
      const std::size_t element = finite_difference_row * row_stride + column;
      const long double value = input[layout.jets.uncheckedValueOffset(
          finite_difference_configuration, element)];
      const long double first = input[layout.jets.uncheckedGradientOffset(
          finite_difference_configuration, dimension, element)];
      const long double trace_second = input[layout.jets.uncheckedLaplacianOffset(
          finite_difference_configuration, 0, element)];
      plus_logits[column] = value + step * first + step * step * trace_second / 6.0L;
      minus_logits[column] = value - step * first + step * step * trace_second / 6.0L;
    }
    plus_probability[dimension] = referenceSoftmax(plus_logits);
    minus_probability[dimension] = referenceSoftmax(minus_logits);
  }
  const std::vector<long double> base_probability = referenceSoftmax(
      {5.0L, -2.0L, 4.5L});
  for (std::size_t column = 0; column < width; ++column)
  {
    const std::size_t element = finite_difference_row * row_stride + column;
    long double trace_second = 0.0L;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      const long double first =
          (plus_probability[dimension][column] - minus_probability[dimension][column]) /
          (2.0L * step);
      checkClose(jets[layout.jets.uncheckedGradientOffset(
                     finite_difference_configuration, dimension, element)],
                 first, 2.0e-8);
      trace_second +=
          (plus_probability[dimension][column] - 2.0L * base_probability[column] +
           minus_probability[dimension][column]) / (step * step);
    }
    checkClose(jets[layout.jets.uncheckedLaplacianOffset(
                   finite_difference_configuration, 0, element)],
               trace_second, 3.0e-7);
  }
}

TEST_CASE("PsiFormer softmax jet host diagnostics reject invalid and non-finite rows",
          "[psiformer][device][spatial]")
{
  CHECK_THROWS_AS(psiformer::makeSoftmaxJetRowLayout(
                      1, 2, 3, 1, psiformer::SpatialJetMode::FULL_VGL, 2),
                  std::invalid_argument);
  CHECK_THROWS_AS(psiformer::makeSoftmaxJetRowLayout(
                      1, 2, 3, 1, psiformer::SpatialJetMode::FULL_VGL, 3, 5),
                  std::invalid_argument);

  const psiformer::SoftmaxJetRowLayout layout = psiformer::makeSoftmaxJetRowLayout(
      1, 1, 3, 1, psiformer::SpatialJetMode::FULL_VGL);
  std::vector<double> jets(psiformer::spatialJetSpanElements(layout.jets), 0.0);
  jets[layout.jets.uncheckedValueOffset(0, 1)] = std::numeric_limits<double>::infinity();
  CHECK_THROWS_AS(psiformer::stableSoftmaxJetRows(layout, jets.data()), std::domain_error);

  std::fill(jets.begin(), jets.end(), 0.0);
  jets[layout.jets.uncheckedGradientOffset(0, 2, 0)] =
      std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS_AS(psiformer::stableSoftmaxJetRows(layout, jets.data()), std::domain_error);

  const psiformer::SoftmaxJetRowLayout active = psiformer::makeSoftmaxJetRowLayout(
      1, 1, 2, 3, psiformer::SpatialJetMode::ACTIVE);
  std::vector<double> active_jets(psiformer::spatialJetSpanElements(active.jets), 0.0);
  active_jets[active.jets.uncheckedValueOffset(0, 0)] = 3.0;
  active_jets[active.jets.uncheckedValueOffset(0, 1)] = -2.0;
  psiformer::stableSoftmaxJetRows(active, active_jets.data());
  CHECK(active_jets[active.jets.uncheckedValueOffset(0, 0)] > 0.99);
}

int main(int argc, char* argv[])
{
  return Catch::Session().run(argc, argv);
}
