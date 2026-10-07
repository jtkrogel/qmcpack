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
#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialJetKernels.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOpenSpatial.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace psiformer = qmcplusplus::psiformer;
namespace math = qmcplusplus::psiformer::device_math;
namespace spatial_jet = qmcplusplus::psiformer::spatial_jet;
namespace open_spatial = qmcplusplus::psiformer::open_spatial;
namespace determinant = qmcplusplus::psiformer::device_determinant;

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

TEST_CASE("PsiFormer active spatial dense and QKV preserve every padded plane",
          "[psiformer][device][spatial]")
{
  constexpr std::size_t configurations = 2;
  constexpr std::size_t rows = 2;
  constexpr std::size_t input_width = 3;
  constexpr std::size_t output_width = 4;
  const psiformer::SpatialDenseJetLayout layout = psiformer::makeSpatialDenseJetLayout(
      configurations, rows, input_width, output_width, 3,
      psiformer::SpatialJetMode::ACTIVE, 4, 5, 5, 9, 11, 38, 47);
  CHECK(layout.source.gradient_lanes == 3);
  CHECK(layout.source.laplacian_lanes == 0);
  CHECK(layout.source.configuration_count == 2);

  std::vector<double> source(psiformer::spatialJetSpanElements(layout.source), -71.0);
  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
    for (std::size_t plane = 0; plane < layout.source.plane_count; ++plane)
      for (std::size_t row = 0; row < rows; ++row)
        for (std::size_t input = 0; input < input_width; ++input)
          source[layout.source.uncheckedPlaneOffset(
              configuration, plane, row * layout.source_row_stride + input)] =
              0.1 * static_cast<double>(1 + input + 3 * row + 7 * plane + 19 * configuration);

  std::array<std::vector<double>, 3> weights{
      std::vector<double>(14), std::vector<double>(14), std::vector<double>(14)};
  for (std::size_t projection = 0; projection < weights.size(); ++projection)
    for (std::size_t input = 0; input < input_width; ++input)
      for (std::size_t output = 0; output < output_width; ++output)
        weights[projection][input * layout.weight_row_stride + output] =
            0.03 * static_cast<double>(1 + output + 2 * input + 5 * projection);

  std::array<std::vector<double>, 3> projected{
      std::vector<double>(psiformer::spatialJetSpanElements(layout.target), -83.0),
      std::vector<double>(psiformer::spatialJetSpanElements(layout.target), -83.0),
      std::vector<double>(psiformer::spatialJetSpanElements(layout.target), -83.0)};
  spatial_jet::projectQkvJetsHost(
      layout, source.data(), weights[0].data(), weights[1].data(), weights[2].data(),
      projected[0].data(), projected[1].data(), projected[2].data());

  for (std::size_t projection = 0; projection < projected.size(); ++projection)
    for (std::size_t configuration = 0; configuration < configurations; ++configuration)
      for (std::size_t plane = 0; plane < layout.source.plane_count; ++plane)
        for (std::size_t row = 0; row < rows; ++row)
          for (std::size_t output = 0; output < output_width; ++output)
          {
            long double expected = 0.0L;
            for (std::size_t input = 0; input < input_width; ++input)
              expected += source[layout.source.uncheckedPlaneOffset(
                              configuration, plane,
                              row * layout.source_row_stride + input)] *
                  static_cast<long double>(weights[projection][
                      input * layout.weight_row_stride + output]);
            checkClose(projected[projection][layout.target.uncheckedPlaneOffset(
                           configuration, plane,
                           row * layout.target_row_stride + output)],
                       expected);
          }

  CHECK_THROWS_AS(psiformer::makeSpatialDenseJetLayout(
                      1, 2, 3, 4, 1, psiformer::SpatialJetMode::ACTIVE, 2),
                  std::invalid_argument);
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

TEST_CASE("PsiFormer full-VGL attention jets match independent context finite differences",
          "[psiformer][device][spatial]")
{
  constexpr std::size_t configurations = 2;
  constexpr std::size_t rows            = 2;
  constexpr std::size_t heads           = 2;
  constexpr std::size_t head_width      = 2;
  constexpr std::size_t feature_width   = heads * head_width;
  const psiformer::SpatialAttentionJetLayout layout =
      psiformer::makeSpatialAttentionJetLayout(
          configurations, rows, heads, head_width, 1,
          psiformer::SpatialJetMode::FULL_VGL,
          5, 3, 7, 11, 14, 57, 73);
  const psiformer::SoftmaxJetRowLayout softmax_layout =
      psiformer::makeAttentionSoftmaxJetRowLayout(layout);
  CHECK(softmax_layout.row_count == heads * rows);
  CHECK(softmax_layout.rows_per_group == rows);
  CHECK(softmax_layout.row_group_stride == 7);

  const std::size_t feature_storage = psiformer::spatialJetSpanElements(layout.features);
  std::vector<double> query(feature_storage, -61.0);
  std::vector<double> key(feature_storage, -62.0);
  std::vector<double> value(feature_storage, -63.0);
  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t feature = 0; feature < feature_width; ++feature)
      {
        const std::size_t element = row * layout.feature_row_stride + feature;
        const double label = static_cast<double>(1 + feature + 5 * row + 13 * configuration);
        query[layout.features.uncheckedValueOffset(configuration, element)] = 0.07 * label - 0.3;
        key[layout.features.uncheckedValueOffset(configuration, element)] = -0.04 * label + 0.5;
        value[layout.features.uncheckedValueOffset(configuration, element)] = 0.05 * label - 0.2;
        for (std::size_t lane = 0; lane < 3; ++lane)
        {
          query[layout.features.uncheckedGradientOffset(configuration, lane, element)] =
              0.01 * label - 0.015 * static_cast<double>(lane);
          key[layout.features.uncheckedGradientOffset(configuration, lane, element)] =
              -0.008 * label + 0.012 * static_cast<double>(lane);
          value[layout.features.uncheckedGradientOffset(configuration, lane, element)] =
              0.006 * label + 0.009 * static_cast<double>(lane);
        }
        query[layout.features.uncheckedLaplacianOffset(configuration, 0, element)] =
            0.013 * label - 0.04;
        key[layout.features.uncheckedLaplacianOffset(configuration, 0, element)] =
            -0.011 * label + 0.03;
        value[layout.features.uncheckedLaplacianOffset(configuration, 0, element)] =
            0.009 * label - 0.02;
      }

  std::vector<double> attention(
      psiformer::spatialJetSpanElements(layout.attention), -75.0);
  std::vector<double> context(feature_storage, -76.0);
  spatial_jet::attentionLogitJetsHost(
      layout, query.data(), key.data(), attention.data());
  psiformer::stableSoftmaxJetRows(softmax_layout, attention.data());
  spatial_jet::attentionContextJetsHost(
      layout, attention.data(), value.data(), context.data());

  const auto reference_context = [&](std::size_t configuration,
                                     std::size_t dimension,
                                     long double displacement) {
    std::vector<long double> output(rows * feature_width, 0.0L);
    const auto component = [&](const std::vector<double>& buffer,
                               std::size_t element) {
      const long double base = buffer[
          layout.features.uncheckedValueOffset(configuration, element)];
      const long double first = buffer[
          layout.features.uncheckedGradientOffset(configuration, dimension, element)];
      const long double trace_second = buffer[
          layout.features.uncheckedLaplacianOffset(configuration, 0, element)];
      return base + displacement * first +
          displacement * displacement * trace_second / 6.0L;
    };

    for (std::size_t head = 0; head < heads; ++head)
      for (std::size_t output_row = 0; output_row < rows; ++output_row)
      {
        std::vector<long double> logits(rows, 0.0L);
        for (std::size_t input_row = 0; input_row < rows; ++input_row)
          for (std::size_t feature = 0; feature < head_width; ++feature)
          {
            const std::size_t query_element =
                output_row * layout.feature_row_stride + head * head_width + feature;
            const std::size_t key_element =
                input_row * layout.feature_row_stride + head * head_width + feature;
            logits[input_row] += component(query, query_element) * component(key, key_element) /
                std::sqrt(static_cast<long double>(head_width));
          }
        const std::vector<long double> probability = referenceSoftmax(logits);
        for (std::size_t feature = 0; feature < head_width; ++feature)
          for (std::size_t input_row = 0; input_row < rows; ++input_row)
          {
            const std::size_t value_element =
                input_row * layout.feature_row_stride + head * head_width + feature;
            output[output_row * feature_width + head * head_width + feature] +=
                probability[input_row] * component(value, value_element);
          }
      }
    return output;
  };

  constexpr long double step = 2.0e-4L;
  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
  {
    const std::vector<long double> base = reference_context(configuration, 0, 0.0L);
    std::array<std::vector<long double>, 3> plus;
    std::array<std::vector<long double>, 3> minus;
    for (std::size_t dimension = 0; dimension < 3; ++dimension)
    {
      plus[dimension] = reference_context(configuration, dimension, step);
      minus[dimension] = reference_context(configuration, dimension, -step);
    }
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t feature = 0; feature < feature_width; ++feature)
      {
        const std::size_t packed = row * feature_width + feature;
        const std::size_t element = row * layout.feature_row_stride + feature;
        checkClose(context[layout.features.uncheckedValueOffset(configuration, element)],
                   base[packed]);
        long double trace_second = 0.0L;
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
        {
          const long double first =
              (plus[dimension][packed] - minus[dimension][packed]) /
              (2.0L * step);
          checkClose(context[layout.features.uncheckedGradientOffset(
                         configuration, dimension, element)],
                     first, 3.0e-8);
          trace_second +=
              (plus[dimension][packed] - 2.0L * base[packed] +
               minus[dimension][packed]) / (step * step);
        }
        checkClose(context[layout.features.uncheckedLaplacianOffset(
                       configuration, 0, element)],
                   trace_second, 3.0e-6);
      }
  }

  CHECK_THROWS_AS(psiformer::makeSpatialAttentionJetLayout(
                      1, 2, 2, 2, 1, psiformer::SpatialJetMode::ACTIVE, 3),
                  std::invalid_argument);
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

TEST_CASE("PsiFormer open raw feature jets match independent finite differences",
          "[psiformer][device][spatial][open]")
{
  CHECK_THROWS_AS(psiformer::makeOpenFeatureJetLayout(
                      1, 2, 0, 1, psiformer::SpatialJetMode::ACTIVE),
                  std::invalid_argument);
  constexpr std::size_t configurations = 2;
  constexpr std::size_t electrons = 2;
  const auto layout = psiformer::makeOpenFeatureJetLayout(
      configurations, electrons, 1, 1, psiformer::SpatialJetMode::FULL_VGL,
      4, 9, 4, 13, 120);
  std::vector<double> positions{
      0.7, -0.2, 0.4, 91.0, -0.3, 0.8, 1.1, 92.0, 93.0,
      1.2, 0.1, -0.6, 94.0, 0.2, -1.0, 0.5, 95.0};
  const std::array<double, 4> nuclei{0.1, -0.4, 0.2, 88.0};
  std::vector<double> output(psiformer::spatialJetSpanElements(layout.output), -17.0);
  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
    CHECK(open_spatial::buildOpenFeatureJetsForConfiguration(
              layout, positions.data(), nuclei.data(), configuration,
              configuration, output.data()) == psiformer::OpenSpatialStatus::REGULAR);

  auto value = [&](std::size_t configuration, std::size_t electron,
                   std::size_t feature, std::size_t moved_electron,
                   std::size_t dimension, long double delta) {
    std::array<long double, 3> displacement{};
    for (std::size_t d = 0; d < 3; ++d)
      displacement[d] = positions[configuration * 9 + electron * 4 + d] - nuclei[d] +
          (electron == moved_electron && d == dimension ? delta : 0.0L);
    const long double radius = std::sqrt(displacement[0] * displacement[0] +
                                         displacement[1] * displacement[1] +
                                         displacement[2] * displacement[2]);
    if (feature == 0)
      return std::log1p(radius);
    if (feature < 4)
      return displacement[feature - 1] * std::log1p(radius) / radius;
    return electron == 0 ? 1.0L : -1.0L;
  };
  constexpr long double step = 1.0e-4L;
  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
    for (std::size_t electron = 0; electron < electrons; ++electron)
      for (std::size_t feature = 0; feature < layout.feature_width; ++feature)
      {
        const std::size_t element = electron * layout.feature_width + feature;
        checkClose(output[layout.output.uncheckedValueOffset(configuration, element)],
                   value(configuration, electron, feature, electron, 0, 0.0L));
        long double trace = 0.0L;
        for (std::size_t dimension = 0; dimension < 3; ++dimension)
        {
          const long double plus = value(configuration, electron, feature, electron,
                                         dimension, step);
          const long double minus = value(configuration, electron, feature, electron,
                                          dimension, -step);
          const long double base = value(configuration, electron, feature, electron,
                                         dimension, 0.0L);
          checkClose(output[layout.output.uncheckedGradientOffset(
                         configuration, 3 * electron + dimension, element)],
                     (plus - minus) / (2.0L * step), 2.0e-8);
          trace += (plus - 2.0L * base + minus) / (step * step);
        }
        checkClose(output[layout.output.uncheckedLaplacianOffset(
                       configuration, electron, element)], trace, 4.0e-7);
      }

  const auto active = psiformer::makeOpenFeatureJetLayout(
      configurations, electrons, 1, 1, psiformer::SpatialJetMode::ACTIVE, 4, 9);
  std::vector<double> active_output(psiformer::spatialJetSpanElements(active.output));
  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
    CHECK(open_spatial::buildOpenFeatureJetsForConfiguration(
              active, positions.data(), nuclei.data(), configuration,
              1 - configuration, active_output.data()) ==
          psiformer::OpenSpatialStatus::REGULAR);
  CHECK(active_output[active.output.uncheckedGradientOffset(0, 0, 0)] == 0.0);
  CHECK(std::abs(active_output[active.output.uncheckedGradientOffset(
                     0, 0, active.feature_width)]) > 0.0);

  std::vector<double> coalesced = positions;
  for (std::size_t dimension = 0; dimension < 3; ++dimension)
    coalesced[dimension] = nuclei[dimension];
  CHECK(open_spatial::buildOpenFeatureJetsForConfiguration(
            active, coalesced.data(), nuclei.data(), 0, 0, active_output.data()) ==
        psiformer::OpenSpatialStatus::COALESCENCE);
}

TEST_CASE("PsiFormer open cusp jets and transforms preserve spatial algebra",
          "[psiformer][device][spatial][open]")
{
  CHECK_THROWS_AS(psiformer::makeOpenCuspJetLayout(
                      1, 2, 3, psiformer::SpatialJetMode::ACTIVE),
                  std::invalid_argument);
  constexpr std::size_t configurations = 2;
  constexpr std::size_t electrons = 3;
  const auto cusp_layout = psiformer::makeOpenCuspJetLayout(
      configurations, electrons, 2, psiformer::SpatialJetMode::FULL_VGL, 4, 13,
      2, 25);
  std::vector<double> positions{
      0.2, -0.1, 0.4, 8.0, 1.0, 0.3, -0.2, 8.0, -0.4, 1.1, 0.6, 8.0, 8.0,
      -0.5, 0.2, 0.9, 8.0, 0.6, -0.7, 0.1, 8.0, 1.3, 0.8, -0.4, 8.0};
  std::vector<double> cusp(psiformer::spatialJetSpanElements(cusp_layout.output), -5.0);
  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
    CHECK(open_spatial::buildOpenCuspJetsForConfiguration(
              cusp_layout, positions.data(), configuration, 0, 0.7, 1.1,
              cusp.data()) == psiformer::OpenSpatialStatus::REGULAR);

  auto cusp_value = [&](std::size_t configuration, std::size_t moved_electron,
                        std::size_t dimension, long double delta) {
    long double result = 0.0L;
    for (std::size_t first = 0; first < electrons; ++first)
      for (std::size_t second = first + 1; second < electrons; ++second)
      {
        long double radius_squared = 0.0L;
        for (std::size_t d = 0; d < 3; ++d)
        {
          const long double first_value = positions[configuration * 13 + first * 4 + d] +
              (first == moved_electron && d == dimension ? delta : 0.0L);
          const long double second_value = positions[configuration * 13 + second * 4 + d] +
              (second == moved_electron && d == dimension ? delta : 0.0L);
          const long double difference = first_value - second_value;
          radius_squared += difference * difference;
        }
        const bool same = (first < 2) == (second < 2);
        const long double alpha = same ? 0.7L : 1.1L;
        const long double factor = same ? 0.25L : 0.5L;
        result -= factor * alpha * alpha / (alpha + std::sqrt(radius_squared));
      }
    return result;
  };
  constexpr long double step = 1.0e-4L;
  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
  {
    checkClose(cusp[cusp_layout.output.uncheckedValueOffset(configuration, 0)],
               cusp_value(configuration, 0, 0, 0.0L));
    for (std::size_t electron = 0; electron < electrons; ++electron)
    {
      long double trace = 0.0L;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const long double plus = cusp_value(configuration, electron, dimension, step);
        const long double minus = cusp_value(configuration, electron, dimension, -step);
        const long double base = cusp_value(configuration, electron, dimension, 0.0L);
        checkClose(cusp[cusp_layout.output.uncheckedGradientOffset(
                       configuration, 3 * electron + dimension, 0)],
                   (plus - minus) / (2.0L * step), 2.0e-8);
        trace += (plus - 2.0L * base + minus) / (step * step);
      }
      checkClose(cusp[cusp_layout.output.uncheckedLaplacianOffset(
                     configuration, electron, 0)], trace, 3.0e-7);
    }
  }

  const auto transform = psiformer::makeSpatialElementwiseJetLayout(
      1, 1, 2, 1, psiformer::SpatialJetMode::FULL_VGL, 3, 4, 21);
  std::vector<double> input(psiformer::spatialJetSpanElements(transform.jets), 0.0);
  std::vector<double> tanh_output(input.size(), 0.0), residual(input.size(), 0.0);
  const std::array<double, 2> bias{0.2, -0.1};
  for (std::size_t element = 0; element < 2; ++element)
  {
    input[transform.jets.uncheckedValueOffset(0, element)] = 0.3 + element;
    for (std::size_t lane = 0; lane < 3; ++lane)
      input[transform.jets.uncheckedGradientOffset(0, lane, element)] = 0.1 * (lane + 1);
    input[transform.jets.uncheckedLaplacianOffset(0, 0, element)] = -0.4;
    CHECK(open_spatial::biasTanhJetElement(transform, input.data(), bias.data(),
                                           0, element, tanh_output.data()) ==
          psiformer::OpenSpatialStatus::REGULAR);
    CHECK(open_spatial::residualJetElement(transform, input.data(), tanh_output.data(),
                                            0, element, residual.data()) ==
          psiformer::OpenSpatialStatus::REGULAR);
    checkClose(tanh_output[transform.jets.uncheckedValueOffset(0, element)],
               std::tanh(static_cast<long double>(0.3 + element + bias[element])));
    checkClose(residual[transform.jets.uncheckedValueOffset(0, element)],
               0.3L + element + std::tanh(static_cast<long double>(0.3 + element + bias[element])));
  }
}

TEST_CASE("PsiFormer final spatial combination distinguishes log Laplacian and ratio",
          "[psiformer][device][spatial][open]")
{
  const auto layout = psiformer::makeFinalSpatialJetLayout(
      2, 2, psiformer::SpatialJetMode::FULL_VGL, 8, 3, 4, 2, 24);
  std::array<determinant::CombinationMetadata, 2> metadata{};
  std::array<determinant::DerivativeStatus, 2> derivative_status{
      determinant::DerivativeStatus::AVAILABLE,
      determinant::DerivativeStatus::AVAILABLE};
  for (std::size_t configuration = 0; configuration < 2; ++configuration)
  {
    metadata[configuration].status = determinant::CombinationStatus::REGULAR;
    metadata[configuration].phase = configuration == 0 ? 1.0 : -1.0;
    metadata[configuration].log_abs = 0.7 + configuration;
  }
  std::array<double, 16> determinant_gradient{};
  std::array<double, 6> determinant_laplacian{};
  for (std::size_t configuration = 0; configuration < 2; ++configuration)
  {
    for (std::size_t lane = 0; lane < 6; ++lane)
      determinant_gradient[configuration * 8 + lane] = 0.1 * (1 + lane + 8 * configuration);
    determinant_laplacian[configuration * 3] = -0.3 - configuration;
    determinant_laplacian[configuration * 3 + 1] = 0.2 + configuration;
  }
  std::vector<double> cusp(psiformer::spatialJetSpanElements(layout.output), 0.0);
  for (std::size_t configuration = 0; configuration < 2; ++configuration)
  {
    cusp[layout.output.uncheckedValueOffset(configuration, 0)] = -0.25;
    for (std::size_t lane = 0; lane < 6; ++lane)
      cusp[layout.output.uncheckedGradientOffset(configuration, lane, 0)] = -0.01 * (lane + 1);
    cusp[layout.output.uncheckedLaplacianOffset(configuration, 0, 0)] = 0.05;
    cusp[layout.output.uncheckedLaplacianOffset(configuration, 1, 0)] = -0.07;
  }
  std::array<double, 2> phase{};
  std::vector<double> output(psiformer::spatialJetSpanElements(layout.output), -9.0);
  std::array<double, 8> ratio{};
  for (std::size_t configuration = 0; configuration < 2; ++configuration)
  {
    CHECK(open_spatial::combineFinalSpatialForConfiguration(
              layout, metadata.data(), derivative_status.data(),
              determinant_gradient.data(), determinant_laplacian.data(), cusp.data(),
              configuration, phase.data(), output.data(), ratio.data()) ==
          psiformer::OpenSpatialStatus::REGULAR);
    CHECK(phase[configuration] == metadata[configuration].phase);
    checkClose(output[layout.output.uncheckedValueOffset(configuration, 0)],
               metadata[configuration].log_abs - 0.25L);
    for (std::size_t electron = 0; electron < 2; ++electron)
    {
      const long double lap_log = determinant_laplacian[configuration * 3 + electron] +
          (electron == 0 ? 0.05L : -0.07L);
      long double square = 0.0L;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const std::size_t lane = 3 * electron + dimension;
        const long double gradient = determinant_gradient[configuration * 8 + lane] -
            0.01L * (lane + 1);
        square += gradient * gradient;
      }
      checkClose(output[layout.output.uncheckedLaplacianOffset(configuration, electron, 0)],
                 lap_log);
      checkClose(ratio[configuration * 4 + electron], lap_log + square);
    }
  }
  metadata[1].status = determinant::CombinationStatus::NODE;
  CHECK(open_spatial::combineFinalSpatialForConfiguration(
            layout, metadata.data(), derivative_status.data(), determinant_gradient.data(),
            determinant_laplacian.data(), cusp.data(), 1, phase.data(), output.data(),
            ratio.data()) == psiformer::OpenSpatialStatus::NODE);
  CHECK(phase[1] == 0.0);
  CHECK(ratio[4] == 0.0);
  CHECK(ratio[5] == 0.0);
  metadata[1].status = determinant::CombinationStatus::REGULAR;
  derivative_status[1] = determinant::DerivativeStatus::REQUIRED_INVERSE_UNAVAILABLE;
  CHECK(open_spatial::combineFinalSpatialForConfiguration(
            layout, metadata.data(), derivative_status.data(), determinant_gradient.data(),
            determinant_laplacian.data(), cusp.data(), 1, phase.data(), output.data(),
            ratio.data()) == psiformer::OpenSpatialStatus::REQUIRED_INVERSE_UNAVAILABLE);
}

int main(int argc, char* argv[])
{
  return Catch::Session().run(argc, argv);
}
