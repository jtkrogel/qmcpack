//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_device_derivative_math.cpp
 * @brief Independent CPU tests for portable PsiFormer score-reverse scalar cores.
 */

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"

#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceDerivativeMath.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace derivative = qmcplusplus::psiformer::device_derivative;
namespace math       = qmcplusplus::psiformer::device_math;

namespace
{

void checkClose(double actual, long double expected, double tolerance = 5.0e-12)
{
  CHECK(actual == Catch::Approx(static_cast<double>(expected)).epsilon(tolerance).margin(tolerance));
}

template<typename Function>
long double centralDifference(const Function& function, long double argument, long double step)
{
  return (function(argument + step) - function(argument - step)) / (2.0L * step);
}

long double softmaxObjective(const std::vector<long double>& logits,
                             const std::vector<double>& adjoints)
{
  const long double maximum = *std::max_element(logits.begin(), logits.end());
  std::vector<long double> exponentials(logits.size());
  long double normalization = 0.0L;
  for (std::size_t index = 0; index < logits.size(); ++index)
  {
    exponentials[index] = std::exp(logits[index] - maximum);
    normalization += exponentials[index];
  }

  long double objective = 0.0L;
  for (std::size_t index = 0; index < logits.size(); ++index)
    objective += static_cast<long double>(adjoints[index]) * exponentials[index] / normalization;
  return objective;
}

long double envelopeOracle(long double radius, long double pi, long double zeta)
{
  return pi * std::exp(-std::abs(zeta * radius));
}

long double cuspOracle(long double radius, long double alpha, long double factor)
{
  return -factor * alpha * alpha / (alpha + radius);
}

} // namespace

TEST_CASE("PsiFormer tanh score reverse matches long-double finite differences",
          "[psiformer][device][derivative]")
{
  for (const std::array<double, 2>& sample :
       {std::array<double, 2>{-2.4, 0.7}, {0.0, -1.1}, {1.3, 2.0}})
  {
    const double input    = sample[0];
    const double upstream = sample[1];
    const double output   = std::tanh(input);
    const auto objective = [upstream](long double value) {
      return static_cast<long double>(upstream) * std::tanh(value);
    };
    const long double expected = centralDifference(objective, input, 1.0e-6L);
    checkClose(derivative::tanhAdjointFromOutput(output, upstream), expected, 2.0e-10);
  }
}

TEST_CASE("PsiFormer softmax row reverse matches an independent probability objective",
          "[psiformer][device][derivative]")
{
  const std::vector<long double> logits{-1.7L, 0.2L, 2.3L, 0.9L};
  const std::vector<double> adjoints{0.4, -1.2, 0.7, 2.0};
  const long double maximum = *std::max_element(logits.begin(), logits.end());
  std::vector<double> probabilities(logits.size());
  long double normalization = 0.0L;
  for (long double logit : logits)
    normalization += std::exp(logit - maximum);
  for (std::size_t index = 0; index < logits.size(); ++index)
    probabilities[index] = static_cast<double>(std::exp(logits[index] - maximum) / normalization);

  double row_reduction = 0.0;
  for (std::size_t index = 0; index < logits.size(); ++index)
    row_reduction += derivative::softmaxRowReductionContribution(
        probabilities[index], adjoints[index]);

  double adjoint_sum = 0.0;
  for (std::size_t differentiated = 0; differentiated < logits.size(); ++differentiated)
  {
    const auto objective = [&](long double varied_logit) {
      std::vector<long double> varied = logits;
      varied[differentiated]          = varied_logit;
      return softmaxObjective(varied, adjoints);
    };
    const long double expected =
        centralDifference(objective, logits[differentiated], 1.0e-6L);
    const double actual = derivative::softmaxLogitAdjoint(
        probabilities[differentiated], adjoints[differentiated], row_reduction);
    checkClose(actual, expected, 3.0e-10);
    adjoint_sum += actual;
  }
  checkClose(adjoint_sum, 0.0L, 3.0e-15);
}

TEST_CASE("PsiFormer scaled QK scalar reverse matches dot-product finite differences",
          "[psiformer][device][derivative]")
{
  constexpr double query         = -0.7;
  constexpr double key           = 1.2;
  constexpr double logit_adjoint = 0.45;
  const double scale             = 1.0 / std::sqrt(5.0);
  const auto query_objective = [](long double varied_query) {
    return static_cast<long double>(logit_adjoint) * varied_query * key /
        std::sqrt(5.0L);
  };
  const auto key_objective = [](long double varied_key) {
    return static_cast<long double>(logit_adjoint) * query * varied_key /
        std::sqrt(5.0L);
  };

  checkClose(derivative::scaledQkAdjointContribution(logit_adjoint, key, scale),
             centralDifference(query_objective, query, 1.0e-6L), 2.0e-11);
  checkClose(derivative::scaledQkAdjointContribution(logit_adjoint, query, scale),
             centralDifference(key_objective, key, 1.0e-6L), 2.0e-11);
}

TEST_CASE("PsiFormer attention-context product reverse matches finite differences",
          "[psiformer][device][derivative]")
{
  constexpr double attention = 0.37;
  constexpr double value     = -1.4;
  constexpr double upstream  = 0.8;
  const auto contributions =
      derivative::attentionContextReverseContributions(attention, value, upstream);
  const auto attention_objective = [](long double varied_attention) {
    return static_cast<long double>(upstream) * varied_attention * value;
  };
  const auto value_objective = [](long double varied_value) {
    return static_cast<long double>(upstream) * attention * varied_value;
  };

  checkClose(contributions.attention_adjoint,
             centralDifference(attention_objective, attention, 1.0e-6L), 2.0e-11);
  checkClose(contributions.value_adjoint,
             centralDifference(value_objective, value, 1.0e-6L), 2.0e-11);
}

TEST_CASE("PsiFormer envelope value and parameter partials match long-double oracles",
          "[psiformer][device][derivative]")
{
  constexpr double radius = 2.3;
  constexpr double pi     = -1.3;
  for (double zeta : {-0.4, 0.6})
  {
    const auto result = derivative::envelopeValueAndPartials(radius, pi, zeta);
    checkClose(result.value, envelopeOracle(radius, pi, zeta));
    const auto pi_objective = [=](long double varied_pi) {
      return envelopeOracle(radius, varied_pi, zeta);
    };
    const auto zeta_objective = [=](long double varied_zeta) {
      return envelopeOracle(radius, pi, varied_zeta);
    };
    checkClose(result.pi_partial, centralDifference(pi_objective, pi, 1.0e-6L), 2.0e-11);
    checkClose(result.zeta_partial, centralDifference(zeta_objective, zeta, 1.0e-6L), 2.0e-10);
  }

  const auto zero_zeta = derivative::envelopeValueAndPartials(radius, pi, 0.0);
  CHECK(zero_zeta.zeta_partial == 0.0);
  const auto zero_radius = derivative::envelopeValueAndPartials(0.0, pi, -0.4);
  CHECK(zero_radius.zeta_partial == 0.0);
  CHECK(zero_radius.value == pi);
  CHECK(zero_radius.pi_partial == 1.0);
}

TEST_CASE("PsiFormer cusp alpha reverse matches long-double finite differences",
          "[psiformer][device][derivative]")
{
  for (const std::array<double, 3>& sample :
       {std::array<double, 3>{0.2, 0.9, 0.25}, {1.7, 0.4, 0.5}, {8.0, 2.1, 0.25}})
  {
    const double radius = sample[0];
    const double alpha  = sample[1];
    const double factor = sample[2];
    const auto result = derivative::cuspValueAndAlphaPartial(radius, alpha, factor);
    checkClose(result.value, cuspOracle(radius, alpha, factor));
    const auto objective = [=](long double varied_alpha) {
      return cuspOracle(radius, varied_alpha, factor);
    };
    checkClose(result.alpha_partial,
               centralDifference(objective, alpha, 1.0e-6L), 2.0e-10);
  }
}

TEST_CASE("PsiFormer derivative finite helper classifies binary64 values",
          "[psiformer][device][derivative]")
{
  CHECK(math::isFiniteBinary64(0.0));
  CHECK(math::isFiniteBinary64(-0.0));
  CHECK(math::isFiniteBinary64(std::numeric_limits<double>::denorm_min()));
  CHECK(math::isFiniteBinary64(std::numeric_limits<double>::max()));
  CHECK_FALSE(math::isFiniteBinary64(std::numeric_limits<double>::infinity()));
  CHECK_FALSE(math::isFiniteBinary64(-std::numeric_limits<double>::infinity()));
  CHECK_FALSE(math::isFiniteBinary64(std::numeric_limits<double>::quiet_NaN()));
}

// Keep this oracle executable independent of QMCPACK's accelerator-initializing
// Catch main so it remains runnable in CUDA/HIP builds on driverless hosts.
int main(int argc, char* argv[])
{
  return Catch::Session().run(argc, argv);
}
