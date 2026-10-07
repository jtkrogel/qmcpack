//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_device_mixed_precision.cpp
 * @brief Independent CPU arithmetic model for the Task 27 FP32 value path.
 *
 * These provisional bounds catch indexing and scalar-type wiring errors.  They do
 * not model vendor GEMM ordering, GPU FMA, device denormal handling, or TF32.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerAttention.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerPrecisionPolicy.h"

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

using namespace qmcplusplus::psiformer;

namespace
{

using MixedDense = PsiFormerPrecisionTraits<
    PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
    PsiFormerArithmeticOperation::DENSE_PROJECTION>;
using MixedOrbital = PsiFormerPrecisionTraits<
    PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
    PsiFormerArithmeticOperation::ORBITAL_CONSTRUCTION>;
using MixedDeterminant = PsiFormerPrecisionTraits<
    PsiFormerPrecisionPolicy::FP32_COMPUTE_FP64_REDUCE,
    PsiFormerArithmeticOperation::DETERMINANT_SOLVE>;

static_assert(std::is_same_v<typename MixedDense::output_type, float>);
static_assert(std::is_same_v<typename MixedOrbital::input_type, double>);
static_assert(std::is_same_v<typename MixedDeterminant::input_type, double>);
static_assert(!std::is_convertible_v<typename MixedDense::output_type*,
                                     typename MixedOrbital::input_type*>);
static_assert(!std::is_convertible_v<typename MixedDense::output_type*,
                                     typename MixedDeterminant::input_type*>);

/// Classify binary32 without relying on release-build fast-math semantics.
bool finiteFloat(float value) noexcept
{
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

/// Apply the deliberately simple row-major FP32 dense arithmetic model.
void denseModel(const DenseForwardLayout& layout,
                const float* source,
                const float* weight,
                float* target)
{
  validateDenseForwardLayout(layout);
  for (std::size_t row = 0; row < layout.rows; ++row)
    for (std::size_t output = 0; output < layout.output_width; ++output)
    {
      float sum = 0.0F;
      for (std::size_t input = 0; input < layout.input_width; ++input)
        sum += source[row * layout.source_row_stride + input] *
            weight[input * layout.weight_row_stride + output];
      target[row * layout.target_row_stride + output] = sum;
    }
}

/// Form configuration-local headed logits with FP32 products and accumulation.
void logitsModel(const BatchedAttentionForwardLayout& layout,
                 const float* query,
                 const float* key,
                 float* logits)
{
  validateBatchedAttentionForwardLayout(layout);
  const float scale = 1.0F / std::sqrt(static_cast<float>(layout.attention.head_width));
  for (std::size_t configuration = 0; configuration < layout.configuration_count;
       ++configuration)
    for (std::size_t head = 0; head < layout.attention.heads; ++head)
      for (std::size_t query_row = 0; query_row < layout.attention.rows; ++query_row)
        for (std::size_t key_row = 0; key_row < layout.attention.rows; ++key_row)
        {
          float sum = 0.0F;
          for (std::size_t feature = 0; feature < layout.attention.head_width; ++feature)
            sum += query[layout.featureOffset(configuration, query_row, head, feature)] *
                key[layout.featureOffset(configuration, key_row, head, feature)];
          logits[layout.attentionOffset(configuration, head, query_row, key_row)] =
              scale * sum;
        }
}

/// Apply stable FP32-storage softmax with FP64 maximum/sum and row diagnostics.
void softmaxModel(const BatchedAttentionForwardLayout& layout,
                  float* values,
                  PsiFormerNumericalDiagnostics& diagnostics)
{
  for (std::size_t configuration = 0; configuration < layout.configuration_count;
       ++configuration)
    for (std::size_t head = 0; head < layout.attention.heads; ++head)
      for (std::size_t query = 0; query < layout.attention.rows; ++query)
      {
        double maximum = -std::numeric_limits<double>::max();
        std::uint64_t invalid = 0;
        for (std::size_t key = 0; key < layout.attention.rows; ++key)
        {
          const float value = values[layout.attentionOffset(configuration, head, query, key)];
          if (finiteFloat(value))
            maximum = std::max(maximum, static_cast<double>(value));
          else
            ++invalid;
        }
        if (invalid != 0)
        {
          diagnostics.nonfinite_count += invalid;
          ++diagnostics.invalid_softmax_count;
          for (std::size_t key = 0; key < layout.attention.rows; ++key)
            values[layout.attentionOffset(configuration, head, query, key)] = 0.0F;
          continue;
        }

        double normalization = 0.0;
        for (std::size_t key = 0; key < layout.attention.rows; ++key)
        {
          const std::size_t offset = layout.attentionOffset(configuration, head, query, key);
          values[offset] = static_cast<float>(
              std::exp(static_cast<double>(values[offset]) - maximum));
          normalization += static_cast<double>(values[offset]);
        }
        for (std::size_t key = 0; key < layout.attention.rows; ++key)
        {
          const std::size_t offset = layout.attentionOffset(configuration, head, query, key);
          values[offset] = static_cast<float>(
              static_cast<double>(values[offset]) / normalization);
        }
      }
}

/// Contract attention and values with FP32 products and accumulation.
void contextModel(const BatchedAttentionForwardLayout& layout,
                  const float* attention,
                  const float* value,
                  float* target)
{
  for (std::size_t configuration = 0; configuration < layout.configuration_count;
       ++configuration)
    for (std::size_t row = 0; row < layout.attention.rows; ++row)
      for (std::size_t head = 0; head < layout.attention.heads; ++head)
        for (std::size_t feature = 0; feature < layout.attention.head_width; ++feature)
        {
          float sum = 0.0F;
          for (std::size_t source = 0; source < layout.attention.rows; ++source)
            sum += attention[layout.attentionOffset(configuration, head, row, source)] *
                value[layout.featureOffset(configuration, source, head, feature)];
          target[layout.featureOffset(configuration, row, head, feature)] = sum;
        }
}

/// Return whether an offset is one of the logical values rather than padding.
bool isLogicalOffset(const BatchedValueLayout& layout, std::size_t offset)
{
  for (std::size_t configuration = 0; configuration < layout.configuration_count;
       ++configuration)
    for (std::size_t row = 0; row < layout.rows; ++row)
      if (offset >= layout.offset(configuration, row, 0) &&
          offset <= layout.offset(configuration, row, layout.width - 1))
        return true;
  return false;
}

} // namespace

TEST_CASE("PsiFormer mixed layouts isolate padded B greater than one storage",
          "[psiformer][device][mixed_precision]")
{
  const AttentionForwardLayout attention =
      makeAttentionForwardLayout(3, 2, 2, 7, 5, 17);
  const BatchedAttentionForwardLayout batch =
      makeBatchedAttentionForwardLayout(2, attention, 21, 36);
  CHECK(batch.featureElements() == 39);
  CHECK(batch.attentionElements() == 66);
  CHECK(batch.softmaxRowCount() == 12);
  CHECK(batch.featureOffset(1, 0, 0, 0) == 21);
  CHECK(batch.featureOffset(1, 2, 1, 1) == 38);
  CHECK(batch.attentionOffset(1, 1, 2, 2) == 65);

  const BatchedValueLayout values = makeBatchedValueLayout(2, 3, 4, 7, 23);
  CHECK(values.logicalElements() == 24);
  CHECK(values.storageElements() == 41);
  CHECK(values.offset(1, 2, 3) == 40);
  CHECK_THROWS_AS(makeBatchedValueLayout(2, 3, 4, 3, 0), std::invalid_argument);
  CHECK_THROWS_AS(makeBatchedAttentionForwardLayout(2, attention, 20, 36),
                  std::invalid_argument);
}

TEST_CASE("PsiFormer mixed CPU model preserves B greater than one attention mapping",
          "[psiformer][device][mixed_precision]")
{
  constexpr std::size_t configurations = 2;
  constexpr std::size_t rows = 3;
  constexpr std::size_t width = 4;
  const DenseForwardLayout dense = makeDenseForwardLayout(
      configurations * rows, width, width, 6, 5, 7);
  std::vector<float> source(dense.sourceElements(), 91.0F);
  std::vector<float> query_weight(dense.weightElements(), -73.0F);
  std::vector<float> key_weight(dense.weightElements(), -73.0F);
  std::vector<float> value_weight(dense.weightElements(), -73.0F);
  for (std::size_t row = 0; row < dense.rows; ++row)
    for (std::size_t feature = 0; feature < width; ++feature)
      source[row * dense.source_row_stride + feature] =
          static_cast<float>(0.11 * (row + 1) - 0.07 * feature);
  for (std::size_t input = 0; input < width; ++input)
    for (std::size_t output = 0; output < width; ++output)
    {
      const std::size_t offset = input * dense.weight_row_stride + output;
      query_weight[offset] = static_cast<float>(0.03 * (1 + input + 2 * output));
      key_weight[offset]   = static_cast<float>(-0.02 * (1 + 2 * input - output));
      value_weight[offset] = static_cast<float>(0.04 * (1 - input + output));
    }

  std::vector<float> query(dense.targetElements(), 101.0F);
  std::vector<float> key(dense.targetElements(), 101.0F);
  std::vector<float> value(dense.targetElements(), 101.0F);
  denseModel(dense, source.data(), query_weight.data(), query.data());
  denseModel(dense, source.data(), key_weight.data(), key.data());
  denseModel(dense, source.data(), value_weight.data(), value.data());
  for (std::size_t row = 0; row < dense.rows; ++row)
    for (std::size_t output = 0; output < width; ++output)
    {
      long double expected = 0.0L;
      for (std::size_t input = 0; input < width; ++input)
        expected += static_cast<long double>(source[row * dense.source_row_stride + input]) *
            query_weight[input * dense.weight_row_stride + output];
      CHECK(query[row * dense.target_row_stride + output] ==
            Catch::Approx(static_cast<float>(expected)).margin(2.0e-6));
    }

  const AttentionForwardLayout single = makeAttentionForwardLayout(rows, 2, 2, 7, 5, 17);
  const BatchedAttentionForwardLayout attention =
      makeBatchedAttentionForwardLayout(configurations, single, 21, 36);
  std::vector<float> weights(attention.attentionElements(), 303.0F);
  logitsModel(attention, query.data(), key.data(), weights.data());
  PsiFormerNumericalDiagnostics diagnostics;
  softmaxModel(attention, weights.data(), diagnostics);
  CHECK(diagnostics.nonfinite_count == 0);
  CHECK(diagnostics.invalid_softmax_count == 0);
  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
    for (std::size_t head = 0; head < single.heads; ++head)
      for (std::size_t row = 0; row < rows; ++row)
      {
        double sum = 0.0;
        for (std::size_t key_row = 0; key_row < rows; ++key_row)
          sum += weights[attention.attentionOffset(configuration, head, row, key_row)];
        CHECK(sum == Catch::Approx(1.0).margin(2.0e-7));
      }

  std::vector<float> context(attention.featureElements(), 707.0F);
  contextModel(attention, weights.data(), value.data(), context.data());
  const BatchedValueLayout values = makeBatchedValueLayout(configurations, rows, width, 7, 21);
  const std::vector<float> bias{0.1F, -0.2F, 0.3F, -0.4F};
  std::vector<float> nonlinear(values.storageElements(), 808.0F);
  std::vector<float> residual(values.storageElements(), 909.0F);
  std::vector<double> promoted(values.storageElements(), -1001.0);
  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
    for (std::size_t row = 0; row < rows; ++row)
      for (std::size_t feature = 0; feature < width; ++feature)
      {
        const std::size_t offset = values.offset(configuration, row, feature);
        nonlinear[offset] = std::tanh(context[offset] + bias[feature]);
        residual[offset]  = nonlinear[offset] + query[offset];
        promoted[offset]  = static_cast<double>(residual[offset]);
        CHECK(promoted[offset] == static_cast<double>(residual[offset]));
      }
  for (std::size_t offset = 0; offset < values.storageElements(); ++offset)
    if (!isLogicalOffset(values, offset))
    {
      CHECK(nonlinear[offset] == 808.0F);
      CHECK(residual[offset] == 909.0F);
      CHECK(promoted[offset] == -1001.0);
    }
  CHECK(promoted[values.offset(0, 0, 0)] != promoted[values.offset(1, 0, 0)]);
}

TEST_CASE("PsiFormer mixed softmax uses FP64 reductions and diagnoses invalid rows",
          "[psiformer][device][mixed_precision]")
{
  const AttentionForwardLayout single = makeAttentionForwardLayout(3, 1, 1, 2, 5, 17);
  const BatchedAttentionForwardLayout batch =
      makeBatchedAttentionForwardLayout(2, single, 7, 19);
  std::vector<float> values(batch.attentionElements(), -19.0F);
  values[batch.attentionOffset(0, 0, 0, 0)] = 1000.0F;
  values[batch.attentionOffset(0, 0, 0, 1)] = 999.0F;
  values[batch.attentionOffset(0, 0, 0, 2)] = -1000.0F;
  values[batch.attentionOffset(1, 0, 1, 0)] =
      std::numeric_limits<float>::quiet_NaN();
  values[batch.attentionOffset(1, 0, 1, 1)] =
      std::numeric_limits<float>::infinity();

  PsiFormerNumericalDiagnostics diagnostics;
  softmaxModel(batch, values.data(), diagnostics);
  double ordinary_sum = 0.0;
  for (std::size_t key = 0; key < 3; ++key)
    ordinary_sum += values[batch.attentionOffset(0, 0, 0, key)];
  CHECK(ordinary_sum == Catch::Approx(1.0).margin(2.0e-7));
  CHECK(values[batch.attentionOffset(0, 0, 0, 0)] >
        values[batch.attentionOffset(0, 0, 0, 1)]);
  CHECK(values[batch.attentionOffset(0, 0, 0, 2)] == 0.0F);
  CHECK(diagnostics.nonfinite_count == 2);
  CHECK(diagnostics.invalid_softmax_count == 1);
  for (std::size_t key = 0; key < 3; ++key)
    CHECK(values[batch.attentionOffset(1, 0, 1, key)] == 0.0F);
  CHECK((diagnostics.nonfinite_count != 0 || diagnostics.invalid_softmax_count != 0));
}

TEST_CASE("PsiFormer mixed sensitive islands require the explicit FP64 cast boundary",
          "[psiformer][device][mixed_precision]")
{
  CHECK((std::is_same_v<typename MixedDense::output_type, float>));
  CHECK((std::is_same_v<typename MixedOrbital::input_type, double>));
  CHECK((std::is_same_v<typename MixedDeterminant::input_type, double>));
  CHECK_FALSE((std::is_convertible_v<float*, typename MixedOrbital::input_type*>));
  CHECK_FALSE((std::is_convertible_v<float*, typename MixedDeterminant::input_type*>));
}

int main(int argc, char* argv[])
{
  return Catch::Session().run(argc, argv);
}
