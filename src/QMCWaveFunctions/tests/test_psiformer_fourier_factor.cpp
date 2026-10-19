//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "QMCWaveFunctions/PsiFormer/PsiFormerFourierFactor.h"
#include <limits>

namespace
{
namespace ps = qmcplusplus::psiformer;
using C = ps::FourierFactorJet::Complex;
C contract(const ps::FourierFactorJet& a, const ps::FourierFactorJet& b)
{
  C result = a.value * b.value + a.laplacian * b.laplacian;
  for (std::size_t d = 0; d < 3; ++d)
    result += a.gradient[d] * b.gradient[d];
  return result;
}
void close(C actual, C expected, double tolerance = 1e-10)
{
  CHECK(std::abs(actual - expected) < tolerance);
}
}

TEST_CASE("Fourier factors preserve zero initialization gauge and complete jets", "[psiformer][fourier_factor]")
{
  const std::array<double, 3> position{.31, -.47, .82}, origin{-.12, .21, .09};
  const std::array<std::array<double, 3>, 3> modes{{{0, 0, 0}, {.7, -.2, .9}, {-.4, .8, .3}}};
  std::array<double, 3> coefficients{1, 0, 0};
  auto evaluate = [&](const auto& point) {
    return ps::evaluateFourierFactor(point, origin, modes.data(), coefficients.data(), coefficients.size());
  };
  const auto initial = evaluate(position);
  close(initial.value, 1);
  close(initial.laplacian, 0);
  for (const auto derivative : initial.gradient)
    close(derivative, 0);
  coefficients = {.9, -.37, .23};
  const auto jet = evaluate(position);
  C laplacian{};
  const double step = 2e-4;
  for (std::size_t d = 0; d < 3; ++d)
  {
    auto plus = position, minus = position;
    plus[d] += step;
    minus[d] -= step;
    close(jet.gradient[d], (evaluate(plus).value - evaluate(minus).value) / (2 * step), 2e-9);
    laplacian += (evaluate(plus).value - 2.0 * jet.value + evaluate(minus).value) / (step * step);
  }
  close(jet.laplacian, laplacian, 3e-8);
  for (std::size_t mode = 0; mode < modes.size(); ++mode)
  {
    const auto basis = ps::fourierModeJet(position, origin, modes[mode]);
    coefficients[mode] += step;
    const auto plus = evaluate(position);
    coefficients[mode] -= 2 * step;
    const auto minus = evaluate(position);
    coefficients[mode] += step;
    close(basis.value, (plus.value - minus.value) / (2 * step));
    close(basis.laplacian, (plus.laplacian - minus.laplacian) / (2 * step));
    for (std::size_t d = 0; d < 3; ++d)
      close(basis.gradient[d], (plus.gradient[d] - minus.gradient[d]) / (2 * step));
  }
  const std::array<double, 3> shift{.17, -.28, .32};
  auto shifted_position = position, shifted_origin = origin;
  for (std::size_t d = 0; d < 3; ++d)
  {
    shifted_position[d] += shift[d];
    shifted_origin[d] += shift[d];
  }
  close(ps::evaluateFourierFactor(shifted_position, shifted_origin, modes.data(),
      coefficients.data(), coefficients.size()).value, jet.value);
  CHECK_THROWS_AS(ps::evaluateFourierFactor(position, origin, modes.data(), coefficients.data(), 0),
      std::invalid_argument);
  auto invalid = origin;
  invalid[1] = std::numeric_limits<double>::infinity();
  CHECK_THROWS_AS(ps::fourierModeJet(position, invalid, modes[1]), std::invalid_argument);
}

TEST_CASE("Fourier multiplier transpose is node safe and includes Laplacian cross terms",
          "[psiformer][fourier_factor]")
{
  const ps::FourierFactorJet direction{C(.4, -.3), {C(-.1, .7), C(.8, .2), C(-.5, .6)}, C(.3, -.9)};
  const ps::FourierFactorJet seed{C(.2, .6), {C(.4, -.8), C(-.2, .9), C(.7, .1)}, C(-.3, .5)};
  for (const C value : {C(1, -.2), C(0, 0)})
  {
    const ps::FourierFactorJet factor{value, {C(.6, -.4), C(-.3, .8), C(.2, .7)}, C(-.7, .1)};
    close(contract(ps::multiplyFourierJets(factor, direction), seed),
        contract(direction, ps::pullBackFourierFactor(factor, seed)));
    // Symmetry also supplies real-coefficient pullbacks without dividing at nodes.
    close(contract(ps::multiplyFourierJets(direction, factor), seed),
        contract(direction, ps::pullBackFourierFactor(factor, seed)));
  }
}

TEST_CASE("Fourier factors are simulation periodic in skew cells and compose complete products",
          "[psiformer][fourier_factor]")
{
  const double two_pi = 2 * std::acos(-1.0);
  const std::array<std::array<double, 3>, 3> cell{{{4, 0, 0}, {1, 3, 0}, {-.5, .7, 5}}};
  const double gx = two_pi / 4, gy = -gx / 3;
  const std::array<std::array<double, 3>, 3> modes{{{0, 0, 0},
      {gx, gy, (.5 * gx - .7 * gy) / 5}, {0, two_pi / 3, (-two_pi - .7 * two_pi / 3) / 5}}};
  const std::array<double, 3> position{.29, -.51, .73}, origin{-.17, .23, .11};
  const std::array<double, 3> coefficients{.7, -.4, .2};
  auto evaluate = [&](const auto& point) {
    return ps::evaluateFourierFactor(point, origin, modes.data(), coefficients.data(), coefficients.size());
  };
  const auto factor = evaluate(position);
  for (const auto& vector : cell)
  {
    auto shifted = position;
    for (std::size_t d = 0; d < 3; ++d)
      shifted[d] += vector[d];
    const auto image = evaluate(shifted);
    close(image.value, factor.value);
    close(image.laplacian, factor.laplacian);
    for (std::size_t d = 0; d < 3; ++d)
      close(image.gradient[d], factor.gradient[d]);
  }
  const std::array<double, 3> inner_momentum{.37, -.21, .46};
  auto product_value = [&](const auto& point) {
    return evaluate(point).value * ps::fourierModeJet(point, origin, inner_momentum).value;
  };
  const auto product = ps::multiplyFourierJets(factor, ps::fourierModeJet(position, origin, inner_momentum));
  const double step = 1e-4;
  C laplacian{};
  for (std::size_t d = 0; d < 3; ++d)
  {
    auto plus = position, minus = position;
    plus[d] += step;
    minus[d] -= step;
    close(product.gradient[d], (product_value(plus) - product_value(minus)) / (2 * step), 2e-8);
    laplacian += (product_value(plus) - 2.0 * product.value + product_value(minus)) / (step * step);
  }
  close(product.laplacian, laplacian, 5e-8);
}

TEST_CASE("Fourier release guards reject nonfinite inputs and accumulated output overflow",
          "[psiformer][fourier_factor][fourier_release_guards]")
{
  volatile double infinity = std::numeric_limits<double>::infinity();
  volatile double nan = std::numeric_limits<double>::quiet_NaN();
  const std::array<double, 3> origin{}, point{.2, .3, .4};
  CHECK_THROWS_AS(ps::fourierModeJet({infinity, 0, 0}, origin, {1, 0, 0}), std::invalid_argument);
  CHECK_THROWS_AS(ps::fourierModeJet(point, {0, nan, 0}, {1, 0, 0}), std::invalid_argument);
  CHECK_THROWS_AS(ps::fourierModeJet(point, origin, {0, infinity, 0}), std::invalid_argument);
  CHECK_THROWS_AS(ps::fourierModeJet(point, origin, {1e200, 0, 0}), std::overflow_error);
  const std::array<std::array<double, 3>, 2> zero_modes{};
  const std::array<double, 2> invalid_coefficients{1, nan};
  CHECK_THROWS_AS(ps::evaluateFourierFactor(point, origin, zero_modes.data(), invalid_coefficients.data(), 2),
      std::invalid_argument);
  const std::array<double, 2> overflowing_sum{std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  CHECK_THROWS_AS(ps::evaluateFourierFactor(point, origin, zero_modes.data(), overflowing_sum.data(), 2),
      std::overflow_error);
  const std::array<double, 3> large_mode{1e10, 0, 0};
  const double large_coefficient = 1e300;
  CHECK_THROWS_AS(ps::evaluateFourierFactor(point, origin, &large_mode, &large_coefficient, 1), std::overflow_error);
}
