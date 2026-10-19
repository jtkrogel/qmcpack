//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include <catch2/catch_test_macros.hpp>
#include "Utilities/for_testing/Catch2Approx.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerPolynomialGeometry.h"

namespace
{
const pf::GeometryBoundary polynomial_cell{pf::GeometryBoundaryKind::PERIODIC,
    {{{5, 0, 0}, {.7, 4, 0}, {-.4, .9, 6}}}, {true, true, true}};

pf::GeometryPosition cartesian(const pf::GeometryPosition& fractional)
{
  pf::GeometryPosition result{};
  for (std::size_t d = 0; d < 3; ++d)
    for (std::size_t a = 0; a < 3; ++a)
      result[d] += fractional[a] * polynomial_cell.lattice_vectors[a][d];
  return result;
}

// Independent triangular solve and original metric contraction, not production jet algebra.
std::array<double, 4> polynomial_values(const pf::GeometryPosition& x)
{
  const double pi = std::acos(-1.0), two_pi = 2 * pi;
  pf::GeometryPosition u{};
  u[2] = x[2] / 6; u[1] = (x[1] - .9 * u[2]) / 4; u[0] = (x[0] - .7 * u[1] + .4 * u[2]) / 5;
  pf::GeometryPosition f{}, g{};
  for (std::size_t a = 0; a < 3; ++a)
  {
    const double w = two_pi * (u[a] - std::floor(u[a] + .5));
    f[a] = std::abs(w) * (1 - .25 * std::pow(std::abs(w) / pi, 3));
    g[a] = w * (1 - 1.5 * std::abs(w) / pi + .5 * w * w / (pi * pi));
  }
  std::array<double, 4> result{};
  double squared = 0;
  for (std::size_t a = 0; a < 3; ++a)
    for (std::size_t b = 0; b < 3; ++b)
    {
      double metric = 0;
      for (std::size_t d = 0; d < 3; ++d)
      {
        metric += polynomial_cell.lattice_vectors[a][d] * polynomial_cell.lattice_vectors[b][d] /
            (two_pi * two_pi);
        if (a == b) result[1 + d] += polynomial_cell.lattice_vectors[a][d] * g[a] / two_pi;
      }
      squared += metric * (a == b ? f[a] * f[a] : g[a] * g[b]);
    }
  result[0] = std::sqrt(squared);
  return result;
}
}

TEST_CASE("Polynomial periodic radius and vector jets retain skew reciprocal contractions",
          "[wavefunction][psiformer][polynomial_geometry]")
{
  const std::array<double, 3> nucleus{};
  pf::PsiFormerGeometryCache cache(1, pf::GeometryPositionView::interleaved(nucleus.data(), 1), polynomial_cell);
  const auto x = cartesian({.17, -.23, .31});
  const auto evaluate = [&](const auto& position, int orientation) {
    cache.update(pf::GeometryPositionView::interleaved(position.data(), 1));
    return pf::polynomialPeriodicJet(cache.electronNucleusPairs(), 0, polynomial_cell, orientation);
  };
  for (int orientation : {1, -1})
  {
    auto oriented = x;
    for (double& component : oriented) component *= orientation;
    const auto reference = polynomial_values(oriented);
    const auto jet = evaluate(x, orientation);
    REQUIRE(jet.differentiable);
    CHECK(jet.radius == Catch::Approx(reference[0]).margin(2e-14));
    for (std::size_t d = 0; d < 3; ++d)
      CHECK(jet.vector[d] == Catch::Approx(reference[1 + d]).margin(2e-14));
    for (double step : {1e-4, 5e-5})
    {
      std::array<double, 4> laplacian{};
      for (std::size_t d = 0; d < 3; ++d)
      {
        auto plus = oriented, minus = oriented;
        plus[d] += step; minus[d] -= step;
        const auto p = polynomial_values(plus), m = polynomial_values(minus);
        CHECK((p[0] - m[0]) / (2 * step) == Catch::Approx(jet.radius_gradient[d]).margin(2e-8));
        for (std::size_t component = 0; component < 3; ++component)
          CHECK((p[1 + component] - m[1 + component]) / (2 * step) ==
                Catch::Approx(jet.vector_jacobian[component][d]).margin(2e-8));
        for (std::size_t feature = 0; feature < 4; ++feature)
          laplacian[feature] += (p[feature] - 2 * reference[feature] + m[feature]) / (step * step);
      }
      CHECK(laplacian[0] == Catch::Approx(jet.radius_laplacian).margin(2e-6));
      for (std::size_t component = 0; component < 3; ++component)
        CHECK(laplacian[1 + component] == Catch::Approx(jet.vector_laplacian[component]).margin(2e-6));
    }
    const auto fingerprint = cache.storageFingerprint();
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
      auto image = x;
      for (std::size_t d = 0; d < 3; ++d) image[d] += polynomial_cell.lattice_vectors[axis][d];
      const auto periodic = evaluate(image, orientation);
      CHECK(periodic.radius == Catch::Approx(jet.radius).margin(2e-14));
      CHECK(periodic.radius_laplacian == Catch::Approx(jet.radius_laplacian).margin(2e-13));
      for (std::size_t d = 0; d < 3; ++d)
      {
        CHECK(periodic.vector[d] == Catch::Approx(jet.vector[d]).margin(2e-14));
        CHECK(periodic.radius_gradient[d] == Catch::Approx(jet.radius_gradient[d]).margin(2e-14));
      }
    }
    CHECK(cache.storageFingerprint() == fingerprint);
  }
}

TEST_CASE("Polynomial zero-angle planes are C1 while periodic wrap seams retain second jets",
          "[wavefunction][psiformer][polynomial_geometry]")
{
  const std::array<double, 3> nucleus{};
  pf::PsiFormerGeometryCache cache(1, pf::GeometryPositionView::interleaved(nucleus.data(), 1), polynomial_cell);
  const auto evaluate = [&](const auto& fractional) {
    const auto x = cartesian(fractional);
    cache.update(pf::GeometryPositionView::interleaved(x.data(), 1));
    return pf::polynomialPeriodicJet(cache.electronNucleusPairs(), 0, polynomial_cell);
  };
  CHECK_FALSE(evaluate(pf::GeometryPosition{0, .19, -.24}).differentiable);
  const auto zero = evaluate(pf::GeometryPosition{});
  CHECK_FALSE(zero.differentiable); CHECK(zero.radius == 0);
  const double two_pi = 2 * std::acos(-1.0);
  const pf::GeometryPosition b0{two_pi / 5, -two_pi * .7 / 20, two_pi * (.7 * .9 / 20 + .4 / 5) / 6};
  for (double delta : {1e-5, 1e-6})
  {
    const auto plus = evaluate(pf::GeometryPosition{delta, .19, -.24});
    const auto minus = evaluate(pf::GeometryPosition{-delta, .19, -.24});
    REQUIRE(plus.differentiable); REQUIRE(minus.differentiable);
    double positive = 0, negative = 0, norm = 0;
    for (std::size_t d = 0; d < 3; ++d)
    {
      positive += b0[d] * plus.vector_laplacian[d];
      negative += b0[d] * minus.vector_laplacian[d]; norm += b0[d] * b0[d];
      for (std::size_t derivative = 0; derivative < 3; ++derivative)
        CHECK(plus.vector_jacobian[d][derivative] == Catch::Approx(minus.vector_jacobian[d][derivative]).margin(1e-12));
    }
    CHECK(positive < 0); CHECK(negative > 0);
    CHECK(positive - negative == Catch::Approx((-6 / std::acos(-1.0) + 12 * delta / std::acos(-1.0)) * norm).margin(1e-12));
  }
  const auto upper = evaluate(pf::GeometryPosition{.5, .19, -.24});
  const auto lower = evaluate(pf::GeometryPosition{-.5, .19, -.24});
  REQUIRE(upper.differentiable); REQUIRE(lower.differentiable);
  CHECK(upper.radius == Catch::Approx(lower.radius).margin(2e-14));
  CHECK(upper.radius_laplacian == Catch::Approx(lower.radius_laplacian).margin(2e-13));
}
