//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_device_spatial_reverse.cpp
 * @brief Independent finite-difference tests for full-VGL spatial-jet reverse.
 */

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "QMCWaveFunctions/PsiFormer/PsiFormerSpatialJetReverse.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace pf = qmcplusplus::psiformer;
namespace forward = qmcplusplus::psiformer::spatial_jet;
namespace reverse = qmcplusplus::psiformer::spatial_jet_reverse;

namespace
{

constexpr double sentinel = -813.25;

double datum(std::size_t index, double shift = 0.0)
{
  return shift + 0.17 * std::sin(0.41 * static_cast<double>(index + 1)) +
      0.09 * std::cos(0.23 * static_cast<double>(index + 3));
}

void checkClose(double actual, double expected, double tolerance = 3.0e-7)
{
  CHECK(actual == Catch::Approx(expected).epsilon(tolerance).margin(tolerance));
}

template<class Function>
double finiteDifference(std::vector<double>& values,
                        std::size_t offset,
                        Function&& objective,
                        double step = 2.0e-6)
{
  values[offset] += step;
  const double plus = objective();
  values[offset] -= 2.0 * step;
  const double minus = objective();
  values[offset] += step;
  return (plus - minus) / (2.0 * step);
}

std::vector<std::size_t> logicalElements(const pf::SpatialElementwiseJetLayout& layout)
{
  std::vector<std::size_t> result;
  for (std::size_t row = 0; row < layout.row_count; ++row)
    for (std::size_t column = 0; column < layout.row_width; ++column)
      result.push_back(row * layout.row_stride + column);
  return result;
}

std::vector<std::size_t> logicalOffsets(const pf::SpatialJetLayout& layout,
                                        const std::vector<std::size_t>& elements)
{
  std::vector<std::size_t> result;
  for (std::size_t plane = 0; plane < layout.plane_count; ++plane)
    for (const std::size_t element : elements)
      result.push_back(layout.uncheckedPlaneOffset(0, plane, element));
  return result;
}

double dotOffsets(const std::vector<double>& left,
                  const std::vector<double>& right,
                  const std::vector<std::size_t>& offsets)
{
  double result = 0.0;
  for (const std::size_t offset : offsets)
    result += left[offset] * right[offset];
  return result;
}

void productForward(const pf::SpatialElementwiseJetLayout& layout,
                    const std::vector<double>& left,
                    const std::vector<double>& right,
                    double scale,
                    std::vector<double>& output)
{
  for (const std::size_t element : logicalElements(layout))
  {
    const auto& jets = layout.jets;
    const double a = left[jets.uncheckedValueOffset(0, element)];
    const double b = right[jets.uncheckedValueOffset(0, element)];
    output[jets.uncheckedValueOffset(0, element)] = scale * a * b;
    for (std::size_t lane = 0; lane < jets.gradient_lanes; ++lane)
      output[jets.uncheckedGradientOffset(0, lane, element)] = scale *
          (left[jets.uncheckedGradientOffset(0, lane, element)] * b +
           a * right[jets.uncheckedGradientOffset(0, lane, element)]);
    for (std::size_t electron = 0; electron < jets.laplacian_lanes; ++electron)
    {
      double cross = 0.0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const std::size_t lane = 3 * electron + dimension;
        cross += left[jets.uncheckedGradientOffset(0, lane, element)] *
            right[jets.uncheckedGradientOffset(0, lane, element)];
      }
      output[jets.uncheckedLaplacianOffset(0, electron, element)] = scale *
          (left[jets.uncheckedLaplacianOffset(0, electron, element)] * b +
           2.0 * cross + a *
               right[jets.uncheckedLaplacianOffset(0, electron, element)]);
    }
  }
}

void tanhForward(const pf::SpatialElementwiseJetLayout& layout,
                 const std::vector<double>& input,
                 std::vector<double>& output)
{
  for (const std::size_t element : logicalElements(layout))
  {
    const auto& jets = layout.jets;
    const double x = input[jets.uncheckedValueOffset(0, element)];
    const double y = std::tanh(x);
    const double first = 1.0 - y * y;
    const double second = -2.0 * y * first;
    output[jets.uncheckedValueOffset(0, element)] = y;
    for (std::size_t lane = 0; lane < jets.gradient_lanes; ++lane)
      output[jets.uncheckedGradientOffset(0, lane, element)] = first *
          input[jets.uncheckedGradientOffset(0, lane, element)];
    for (std::size_t electron = 0; electron < jets.laplacian_lanes; ++electron)
    {
      double norm = 0.0;
      for (std::size_t dimension = 0; dimension < 3; ++dimension)
      {
        const double gradient = input[jets.uncheckedGradientOffset(
            0, 3 * electron + dimension, element)];
        norm += gradient * gradient;
      }
      output[jets.uncheckedLaplacianOffset(0, electron, element)] =
          first * input[jets.uncheckedLaplacianOffset(0, electron, element)] +
          second * norm;
    }
  }
}

std::vector<std::size_t> softmaxOffsets(const pf::SoftmaxJetRowLayout& layout)
{
  std::vector<std::size_t> result;
  for (std::size_t plane = 0; plane < layout.jets.plane_count; ++plane)
    for (std::size_t row = 0; row < layout.row_count; ++row)
    {
      const std::size_t group = row / layout.rows_per_group;
      const std::size_t within = row % layout.rows_per_group;
      const std::size_t base = group * layout.row_group_stride +
          within * layout.row_stride;
      for (std::size_t column = 0; column < layout.row_width; ++column)
        result.push_back(plane * layout.jets.plane_stride + base + column);
    }
  return result;
}

std::vector<std::size_t> featureOffsets(const pf::SpatialAttentionJetLayout& layout)
{
  std::vector<std::size_t> elements;
  for (std::size_t row = 0; row < layout.rows; ++row)
    for (std::size_t head = 0; head < layout.heads; ++head)
      for (std::size_t feature = 0; feature < layout.head_width; ++feature)
        elements.push_back(forward::featureElement(layout, row, head, feature));
  return logicalOffsets(layout.features, elements);
}

} // namespace

TEST_CASE("PsiFormer product and tanh full-VGL reverse match finite differences",
          "[psiformer][device][spatial][reverse]")
{
  const auto layout = pf::makeSpatialElementwiseJetLayout(
      1, 2, 3, 2, pf::SpatialJetMode::FULL_VGL, 5, 13, 128);
  const auto elements = logicalElements(layout);
  const auto offsets = logicalOffsets(layout.jets, elements);
  const std::size_t span = pf::spatialJetSpanElements(layout.jets);
  std::vector<double> left(span, sentinel), right(span, sentinel);
  std::vector<double> upstream(span, sentinel);
  for (const std::size_t offset : offsets)
  {
    left[offset] = datum(offset, 0.2);
    right[offset] = datum(offset + 19, -0.1);
    upstream[offset] = datum(offset + 47);
  }
  std::vector<double> output(span, sentinel);
  constexpr double scale = -0.73;
  auto productObjective = [&] {
    std::fill(output.begin(), output.end(), sentinel);
    productForward(layout, left, right, scale, output);
    return dotOffsets(output, upstream, offsets);
  };
  std::vector<double> left_adjoint(span, sentinel), right_adjoint(span, sentinel);
  for (const std::size_t offset : offsets)
    left_adjoint[offset] = right_adjoint[offset] = 0.0;
  reverse::elementwiseProductJetsReverseHost(
      layout, left.data(), right.data(), upstream.data(), scale,
      left_adjoint.data(), right_adjoint.data());
  for (const std::size_t offset : offsets)
  {
    checkClose(left_adjoint[offset], finiteDifference(left, offset, productObjective));
    checkClose(right_adjoint[offset], finiteDifference(right, offset, productObjective));
  }
  CHECK(left_adjoint[3] == sentinel);

  std::vector<double> tanh_output(span, sentinel), input_adjoint(span, sentinel);
  for (const std::size_t offset : offsets)
    input_adjoint[offset] = 0.0;
  tanhForward(layout, left, tanh_output);
  reverse::tanhJetsReverseHost(layout, left.data(), tanh_output.data(),
                               upstream.data(), input_adjoint.data());
  auto tanhObjective = [&] {
    std::fill(tanh_output.begin(), tanh_output.end(), sentinel);
    tanhForward(layout, left, tanh_output);
    return dotOffsets(tanh_output, upstream, offsets);
  };
  for (const std::size_t offset : offsets)
    checkClose(input_adjoint[offset], finiteDifference(left, offset, tanhObjective),
               8.0e-7);
  CHECK(input_adjoint[3] == sentinel);
}

TEST_CASE("PsiFormer dense full-VGL reverse includes every plane and value bias",
          "[psiformer][device][spatial][reverse]")
{
  const auto layout = pf::makeSpatialDenseJetLayout(
      1, 2, 2, 3, 1, pf::SpatialJetMode::FULL_VGL, 4, 5, 6, 11, 15);
  const std::vector<std::size_t> source_elements{0, 1, 4, 5};
  const std::vector<std::size_t> target_elements{0, 1, 2, 6, 7, 8};
  const auto source_offsets = logicalOffsets(layout.source, source_elements);
  const auto target_offsets = logicalOffsets(layout.target, target_elements);
  std::vector<double> source(pf::spatialJetSpanElements(layout.source), sentinel);
  std::vector<double> target_adjoint(pf::spatialJetSpanElements(layout.target), sentinel);
  std::vector<double> weight(layout.input_width * layout.weight_row_stride, sentinel);
  std::vector<double> bias(layout.output_width);
  for (const std::size_t offset : source_offsets)
    source[offset] = datum(offset);
  for (const std::size_t offset : target_offsets)
    target_adjoint[offset] = datum(offset + 71);
  for (std::size_t input = 0; input < layout.input_width; ++input)
    for (std::size_t output = 0; output < layout.output_width; ++output)
      weight[input * layout.weight_row_stride + output] = datum(11 * input + output);
  for (std::size_t output = 0; output < bias.size(); ++output)
    bias[output] = datum(output + 9);

  std::vector<double> source_adjoint(source.size(), sentinel);
  std::vector<double> weight_adjoint(weight.size(), sentinel);
  std::vector<double> bias_adjoint(bias.size());
  for (const auto offset : source_offsets)
    source_adjoint[offset] = 0.0;
  for (std::size_t input = 0; input < layout.input_width; ++input)
    for (std::size_t output = 0; output < layout.output_width; ++output)
      weight_adjoint[input * layout.weight_row_stride + output] = 0.0;
  reverse::denseJetsReverseHost(layout, source.data(), weight.data(),
                                target_adjoint.data(), source_adjoint.data(),
                                weight_adjoint.data(), bias_adjoint.data());
  std::vector<double> target(pf::spatialJetSpanElements(layout.target), sentinel);
  auto objective = [&] {
    std::fill(target.begin(), target.end(), sentinel);
    forward::denseJetsHost(layout, source.data(), weight.data(), target.data());
    for (std::size_t row = 0; row < layout.rows; ++row)
      for (std::size_t output = 0; output < layout.output_width; ++output)
        target[layout.target.uncheckedValueOffset(
            0, row * layout.target_row_stride + output)] += bias[output];
    return dotOffsets(target, target_adjoint, target_offsets);
  };
  for (const std::size_t offset : source_offsets)
    checkClose(source_adjoint[offset], finiteDifference(source, offset, objective));
  for (std::size_t input = 0; input < layout.input_width; ++input)
    for (std::size_t output = 0; output < layout.output_width; ++output)
    {
      const std::size_t offset = input * layout.weight_row_stride + output;
      checkClose(weight_adjoint[offset], finiteDifference(weight, offset, objective));
    }
  for (std::size_t output = 0; output < layout.output_width; ++output)
    checkClose(bias_adjoint[output], finiteDifference(bias, output, objective));
  CHECK(source_adjoint[2] == sentinel);
  CHECK(weight_adjoint[3] == sentinel);
}

TEST_CASE("PsiFormer softmax full-VGL reverse carries the third derivative term",
          "[psiformer][device][spatial][reverse]")
{
  const auto layout = pf::makeSoftmaxJetRowLayout(
      1, 2, 4, 2, pf::SpatialJetMode::FULL_VGL, 6, 17, 0, 1, 13);
  const auto offsets = softmaxOffsets(layout);
  const std::size_t span = pf::spatialJetSpanElements(layout.jets);
  std::vector<double> logits(span, sentinel), upstream(span, sentinel);
  for (const std::size_t offset : offsets)
  {
    logits[offset] = datum(offset, 0.05);
    upstream[offset] = datum(offset + 101);
  }
  std::vector<double> probabilities = logits;
  pf::stableSoftmaxJetRows(layout, probabilities.data());
  std::vector<double> adjoint(span, sentinel);
  for (const std::size_t offset : offsets)
    adjoint[offset] = 0.0;
  reverse::softmaxJetRowsReverseHost(layout, logits.data(), probabilities.data(),
                                     upstream.data(), adjoint.data());
  auto objective = [&] {
    probabilities = logits;
    pf::stableSoftmaxJetRows(layout, probabilities.data());
    return dotOffsets(probabilities, upstream, offsets);
  };
  for (const std::size_t offset : offsets)
    checkClose(adjoint[offset], finiteDifference(logits, offset, objective), 2.0e-6);

  for (std::size_t plane = 0; plane < layout.jets.plane_count; ++plane)
    for (std::size_t row = 0; row < layout.row_count; ++row)
    {
      const std::size_t base = row * layout.row_group_stride;
      double sum = 0.0;
      for (std::size_t column = 0; column < layout.row_width; ++column)
        sum += adjoint[plane * layout.jets.plane_stride + base + column];
      CHECK(std::abs(sum) < 2.0e-12);
    }
  CHECK(adjoint[4] == sentinel);
}

TEST_CASE("PsiFormer padded multihead attention reverse matches end-to-end differences",
          "[psiformer][device][spatial][reverse]")
{
  const auto layout = pf::makeSpatialAttentionJetLayout(
      1, 2, 2, 2, 1, pf::SpatialJetMode::FULL_VGL, 6, 4, 11, 15, 27);
  const auto feature_offsets = featureOffsets(layout);
  const auto softmax_layout = pf::makeAttentionSoftmaxJetRowLayout(layout);
  const auto attention_offsets = softmaxOffsets(softmax_layout);
  std::vector<double> query(pf::spatialJetSpanElements(layout.features), sentinel);
  std::vector<double> key(query.size(), sentinel), value(query.size(), sentinel);
  std::vector<double> context_adjoint(query.size(), sentinel);
  for (const std::size_t offset : feature_offsets)
  {
    query[offset] = datum(offset, 0.1);
    key[offset] = datum(offset + 31, -0.1);
    value[offset] = datum(offset + 67);
    context_adjoint[offset] = datum(offset + 97);
  }
  std::vector<double> logits(pf::spatialJetSpanElements(layout.attention), sentinel);
  std::vector<double> probability(logits.size(), sentinel);
  std::vector<double> context(query.size(), sentinel);
  auto objective = [&] {
    std::fill(logits.begin(), logits.end(), sentinel);
    forward::attentionLogitJetsHost(layout, query.data(), key.data(), logits.data());
    probability = logits;
    pf::stableSoftmaxJetRows(softmax_layout, probability.data());
    std::fill(context.begin(), context.end(), sentinel);
    forward::attentionContextJetsHost(
        layout, probability.data(), value.data(), context.data());
    return dotOffsets(context, context_adjoint, feature_offsets);
  };
  (void)objective();

  std::vector<double> probability_adjoint(probability.size(), sentinel);
  std::vector<double> value_adjoint(value.size(), sentinel);
  std::vector<double> logit_adjoint(logits.size(), sentinel);
  std::vector<double> query_adjoint(query.size(), sentinel);
  std::vector<double> key_adjoint(key.size(), sentinel);
  for (const std::size_t offset : attention_offsets)
    probability_adjoint[offset] = logit_adjoint[offset] = 0.0;
  for (const std::size_t offset : feature_offsets)
    value_adjoint[offset] = query_adjoint[offset] = key_adjoint[offset] = 0.0;
  reverse::attentionContextJetsReverseHost(
      layout, probability.data(), value.data(), context_adjoint.data(),
      probability_adjoint.data(), value_adjoint.data());
  reverse::softmaxJetRowsReverseHost(
      softmax_layout, logits.data(), probability.data(), probability_adjoint.data(),
      logit_adjoint.data());
  reverse::attentionLogitJetsReverseHost(
      layout, query.data(), key.data(), logit_adjoint.data(),
      query_adjoint.data(), key_adjoint.data());

  for (const std::size_t offset : feature_offsets)
  {
    checkClose(query_adjoint[offset], finiteDifference(query, offset, objective), 4.0e-6);
    checkClose(key_adjoint[offset], finiteDifference(key, offset, objective), 4.0e-6);
    checkClose(value_adjoint[offset], finiteDifference(value, offset, objective), 4.0e-6);
  }
  CHECK(query_adjoint[4] == sentinel);
  CHECK(logit_adjoint[2] == sentinel);
}

TEST_CASE("PsiFormer spatial reverse rejects non-full and batched layouts",
          "[psiformer][device][spatial][reverse]")
{
  const auto active = pf::makeSpatialElementwiseJetLayout(
      1, 1, 2, 1, pf::SpatialJetMode::ACTIVE);
  std::vector<double> storage(pf::spatialJetSpanElements(active.jets));
  CHECK_THROWS_AS(reverse::tanhJetsReverseHost(
                      active, storage.data(), storage.data(), storage.data(),
                      storage.data()),
                  std::invalid_argument);
  const auto full = pf::makeSpatialElementwiseJetLayout(
      1, 1, 2, 1, pf::SpatialJetMode::FULL_VGL);
  storage.resize(pf::spatialJetSpanElements(full.jets));
  std::vector<double> distinct(storage.size());
  CHECK_THROWS_AS(reverse::tanhJetsReverseHost(
                      full, storage.data(), distinct.data(), distinct.data(),
                      storage.data()),
                  std::invalid_argument);
  const auto batched = pf::makeSpatialElementwiseJetLayout(
      2, 1, 2, 1, pf::SpatialJetMode::FULL_VGL);
  storage.resize(pf::spatialJetSpanElements(batched.jets));
  CHECK_THROWS_AS(reverse::elementwiseProductJetsReverseHost(
                      batched, storage.data(), storage.data(), storage.data(), 1.0,
                      storage.data(), storage.data()),
                  std::invalid_argument);
}

int main(int argc, char* argv[])
{
  return Catch::Session().run(argc, argv);
}
