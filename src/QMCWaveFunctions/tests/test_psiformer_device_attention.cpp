//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_device_attention.cpp
 * @brief CPU oracle/property tests for portable PsiFormer dense-attention layouts.
 */

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "QMCWaveFunctions/PsiFormer/PsiFormerAttention.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDenseKernels.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

using qmcplusplus::psiformer::AttentionForwardLayout;
using qmcplusplus::psiformer::DenseForwardLayout;

namespace
{

void checkClose(double actual, long double expected, double tolerance = 5.0e-13)
{
  CHECK(actual == Catch::Approx(static_cast<double>(expected)).epsilon(tolerance).margin(tolerance));
}

std::vector<double> referenceAttention(const AttentionForwardLayout& layout,
                                       const std::vector<double>& query,
                                       const std::vector<double>& key)
{
  std::vector<double> attention(layout.attentionElements(), -91.0);
  const long double scale = 1.0L / std::sqrt(static_cast<long double>(layout.head_width));
  for (std::size_t head = 0; head < layout.heads; ++head)
    for (std::size_t query_row = 0; query_row < layout.rows; ++query_row)
    {
      std::vector<long double> logits(layout.rows);
      for (std::size_t key_row = 0; key_row < layout.rows; ++key_row)
        for (std::size_t feature = 0; feature < layout.head_width; ++feature)
          logits[key_row] += scale *
              static_cast<long double>(query[layout.featureOffset(query_row, head, feature)]) *
              key[layout.featureOffset(key_row, head, feature)];
      const long double maximum = *std::max_element(logits.begin(), logits.end());
      long double sum = 0;
      for (long double& logit : logits)
      {
        logit = std::exp(logit - maximum);
        sum += logit;
      }
      for (std::size_t key_row = 0; key_row < layout.rows; ++key_row)
        attention[layout.attentionOffset(head, query_row, key_row)] =
            static_cast<double>(logits[key_row] / sum);
    }
  return attention;
}

} // namespace

TEST_CASE("PsiFormer dense and attention layouts preserve padded row-major storage",
          "[psiformer][device][attention]")
{
  const DenseForwardLayout dense = qmcplusplus::psiformer::makeDenseForwardLayout(3, 5, 7, 8, 9, 10);
  CHECK(dense.sourceElements() == 21);
  CHECK(dense.weightElements() == 43);
  CHECK(dense.targetElements() == 27);

  const AttentionForwardLayout attention =
      qmcplusplus::psiformer::makeAttentionForwardLayout(3, 2, 4, 11, 5, 17);
  CHECK(attention.featureWidth() == 8);
  CHECK(attention.softmaxRowCount() == 6);
  CHECK(attention.featureElements() == 30);
  CHECK(attention.attentionElements() == 30);
  CHECK(attention.featureOffset(2, 1, 3) == 29);
  CHECK(attention.attentionOffset(1, 2, 2) == 29);

  CHECK_THROWS_AS(qmcplusplus::psiformer::makeDenseForwardLayout(0, 2, 2), std::invalid_argument);
  CHECK_THROWS_AS(qmcplusplus::psiformer::makeDenseForwardLayout(2, 3, 4, 2), std::invalid_argument);
  CHECK_THROWS_AS(qmcplusplus::psiformer::makeAttentionForwardLayout(3, 2, 4, 7), std::invalid_argument);
  CHECK_THROWS_AS(qmcplusplus::psiformer::makeAttentionForwardLayout(3, 2, 4, 8, 2), std::invalid_argument);
  CHECK_THROWS_AS(qmcplusplus::psiformer::makeDenseForwardLayout(
                      static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1, 2, 2),
                  std::length_error);
}

TEST_CASE("PsiFormer stable softmax handles adversarial finite rows",
          "[psiformer][device][attention]")
{
  for (const std::vector<double>& logits :
       {std::vector<double>{1000.0, 999.0, -1000.0},
        std::vector<double>{-800.0, -800.0, -800.0, -800.0},
        std::vector<double>{42.0},
        std::vector<double>{-12.0, 0.25, 9.0, 9.0, 2.0}})
  {
    std::vector<double> weights(logits.size());
    qmcplusplus::psiformer::stableSoftmaxRow(logits.data(), logits.size(), weights.data());
    const long double maximum = *std::max_element(logits.begin(), logits.end());
    long double denominator = 0;
    for (double logit : logits)
      denominator += std::exp(static_cast<long double>(logit) - maximum);
    double sum = 0;
    for (std::size_t column = 0; column < logits.size(); ++column)
    {
      checkClose(weights[column],
                 std::exp(static_cast<long double>(logits[column]) - maximum) / denominator);
      CHECK(weights[column] >= 0.0);
      CHECK(std::isfinite(weights[column]));
      sum += weights[column];
    }
    checkClose(sum, 1.0L);
  }
  CHECK_THROWS_AS(qmcplusplus::psiformer::stableSoftmaxRow(nullptr, 0, nullptr), std::invalid_argument);
  std::array<double, 2> output{};
  const std::array<double, 2> positive_infinity{0.0, std::numeric_limits<double>::infinity()};
  const std::array<double, 2> negative_infinity{0.0, -std::numeric_limits<double>::infinity()};
  const std::array<double, 2> not_a_number{0.0, std::numeric_limits<double>::quiet_NaN()};
  CHECK_THROWS_AS(qmcplusplus::psiformer::stableSoftmaxRow(
                      positive_infinity.data(), positive_infinity.size(), output.data()),
                  std::domain_error);
  CHECK_THROWS_AS(qmcplusplus::psiformer::stableSoftmaxRow(
                      negative_infinity.data(), negative_infinity.size(), output.data()),
                  std::domain_error);
  CHECK_THROWS_AS(qmcplusplus::psiformer::stableSoftmaxRow(
                      not_a_number.data(), not_a_number.size(), output.data()),
                  std::domain_error);
}

TEST_CASE("PsiFormer dense QKV layout matches independent row-major products",
          "[psiformer][device][attention]")
{
  constexpr std::size_t rows  = 3;
  constexpr std::size_t width = 4;
  const DenseForwardLayout layout =
      qmcplusplus::psiformer::makeDenseForwardLayout(rows, width, width);
  const std::vector<double> source{0.3, -0.5, 1.0, 0.2,
                                   -0.7, 0.4, 0.1, 0.8,
                                   1.2, 0.6, -0.9, -0.3};
  std::array<std::vector<double>, 3> weights{
      std::vector<double>{0.2, -0.3, 0.5, 0.7,  -0.1, 0.4, 0.8, -0.6,
                          0.9, 0.2, -0.5, 0.1,  0.3, -0.8, 0.6, 0.4},
      std::vector<double>{-0.4, 0.1, 0.7, -0.2,  0.6, 0.5, -0.3, 0.8,
                          0.2, -0.9, 0.4, 0.3,  0.1, 0.7, -0.6, 0.5},
      std::vector<double>{0.8, 0.2, -0.1, 0.4,  -0.5, 0.9, 0.3, -0.7,
                          0.6, -0.2, 0.5, 0.1,  -0.3, 0.4, 0.7, 0.2}};
  std::array<std::vector<double>, 3> projected{
      std::vector<double>(layout.targetElements()),
      std::vector<double>(layout.targetElements()),
      std::vector<double>(layout.targetElements())};
  qmcplusplus::psiformer::dense::projectQkvReal(
      source.data(), weights[0].data(), weights[1].data(), weights[2].data(),
      rows, width, projected[0].data(), projected[1].data(), projected[2].data());

  for (std::size_t projection = 0; projection < projected.size(); ++projection)
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t output = 0; output < width; ++output)
      {
        long double expected = 0;
        for (std::size_t input = 0; input < width; ++input)
          expected += static_cast<long double>(source[row * width + input]) *
              weights[projection][input * width + output];
        checkClose(projected[projection][row * width + output], expected);
      }
}

TEST_CASE("PsiFormer attention layout matches the established CPU forward oracle",
          "[psiformer][device][attention]")
{
  constexpr std::size_t rows       = 3;
  constexpr std::size_t heads      = 2;
  constexpr std::size_t head_width = 2;
  constexpr std::size_t width      = heads * head_width;
  const AttentionForwardLayout layout =
      qmcplusplus::psiformer::makeAttentionForwardLayout(rows, heads, head_width);
  const std::vector<double> query{0.3, -0.7, 1.1, 0.2,
                                  -0.4, 0.9, 0.5, -1.3,
                                  1.2, 0.1, -0.8, 0.6};
  const std::vector<double> key{-0.2, 0.4, 0.7, -0.5,
                                1.0, -0.3, 0.2, 0.8,
                                -0.6, 1.1, -0.9, 0.3};
  const std::vector<double> value{0.5, -0.1, 0.8, 1.2,
                                  -0.7, 0.4, 0.3, -0.2,
                                  0.9, 1.0, -0.6, 0.7};

  std::vector<double> cpu_attention(layout.attentionElements());
  qmcplusplus::psiformer::dense::attentionWeightsReal(
      query.data(), width, key.data(), width, rows, heads, head_width, cpu_attention.data());
  const std::vector<double> reference = referenceAttention(layout, query, key);
  for (std::size_t element = 0; element < reference.size(); ++element)
    checkClose(cpu_attention[element], reference[element]);

  std::vector<double> cpu_context(rows * width);
  qmcplusplus::psiformer::dense::attentionContextReal(
      cpu_attention.data(), value.data(), width, rows, heads, head_width, cpu_context.data());
  for (std::size_t row = 0; row < rows; ++row)
    for (std::size_t head = 0; head < heads; ++head)
      for (std::size_t feature = 0; feature < head_width; ++feature)
      {
        long double expected = 0;
        for (std::size_t source = 0; source < rows; ++source)
          expected += static_cast<long double>(reference[layout.attentionOffset(head, row, source)]) *
              value[layout.featureOffset(source, head, feature)];
        checkClose(cpu_context[layout.featureOffset(row, head, feature)], expected);
      }
}

int main(int argc, char* argv[])
{
  return Catch::Session().run(argc, argv);
}
