//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
using Catch::Approx;
#include "QMCWaveFunctions/FrozenReferenceSumMath.h"

using namespace qmcplusplus::residual;

/// Check derivative numerators in physical units for a modest, safely exponentiable scale.
static void checkPhysical(const ScaledSpatialJet& result, double value, double gradient, double laplacian)
{
  const long double scale = std::exp(result.log_scale);
  CHECK(static_cast<double>(scale * result.jet.value) == Approx(value).margin(1e-13));
  CHECK(static_cast<double>(scale * result.jet.gradient[0]) == Approx(gradient).margin(1e-13));
  CHECK(static_cast<double>(scale * result.jet.laplacian[0]) == Approx(laplacian).margin(1e-13));
}

TEST_CASE("Frozen reference sum preserves derivatives at either child node", "[wavefunction][residual]")
{
  // Phi=x^2 and N=exp(x), at x=0: Phi has a node but the total does not.
  SpatialJet reference(1);
  reference.laplacian[0] = 2;
  SpatialJet neural(1, 1);
  neural.gradient[0] = 1;
  neural.laplacian[0] = 1;
  auto total = sum({0, reference}, {0, neural}, -0.5);
  checkPhysical(total, -0.5, -0.5, 1.5);
  auto log = logarithmic(total);
  CHECK(log.sign == -1);
  CHECK(static_cast<double>(log.gradient[0]) == Approx(1));
  CHECK(static_cast<double>(log.laplacian[0]) == Approx(-4));

  // Phi=1 and N=x, at x=0: the neural node also carries a finite derivative.
  reference = SpatialJet(1, 1);
  neural = SpatialJet(1);
  neural.gradient[0] = 1;
  log = logarithmic(sum({0, reference}, {0, neural}, 2));
  CHECK(static_cast<double>(log.gradient[0]) == Approx(2));
  CHECK(static_cast<double>(log.laplacian[0]) == Approx(-4));
}

TEST_CASE("Frozen reference sum distinguishes child cancellation from total nodes", "[wavefunction][residual]")
{
  SpatialJet branch(1, 1);
  branch.gradient[0] = 3;
  const auto unchanged = sum({10000, branch}, {-10000, branch}, 0);
  CHECK(unchanged.log_scale == 10000);
  CHECK(unchanged.jet.value == 1);
  CHECK(unchanged.jet.gradient[0] == 3);
  CHECK(static_cast<double>(logarithmic(unchanged).log_abs) == Approx(10000));

  // Log scales far outside double amplitude range remain usable without exp(log|Psi|).
  const auto scaled = logarithmic(sum({10000, branch}, {9999, branch}, 0.5));
  CHECK(std::isfinite(scaled.log_abs));
  CHECK(static_cast<double>(scaled.log_abs - 10000) == Approx(std::log1p(0.5 * std::exp(-1.0))));
  const auto zero = sum({0, branch}, {0, branch}, -1);
  CHECK(zero.jet.value == 0);
  CHECK_THROWS_AS(logarithmic(zero), std::domain_error);
  CHECK_THROWS_AS(sum({0, branch}, {0, branch}, std::numeric_limits<long double>::infinity()),
                  std::invalid_argument);
  CHECK_THROWS_AS(sum({0, branch}, {0, SpatialJet(2, 1)}, 1), std::invalid_argument);
}

TEST_CASE("Residual determinant spatial numerators do not require an inverse", "[wavefunction][residual]")
{
  // det([[x,1],[1,x]])=x^2-1 has a simple node at x=1.
  std::vector<SpatialJet> matrix(4, SpatialJet(1, 1));
  matrix[0].gradient[0] = matrix[3].gradient[0] = 1;
  auto det = determinant(matrix, 2);
  checkPhysical(det, 0, 2, 2);
  auto total = logarithmic(sum({0, SpatialJet(1, 4)}, det, 1));
  CHECK(static_cast<double>(total.gradient[0]) == Approx(0.5));
  CHECK(static_cast<double>(total.laplacian[0]) == Approx(0.25));

  // Two zero value rows retain the mixed product derivatives: det(diag(x,x))=x^2.
  matrix.assign(4, SpatialJet(1));
  matrix[0].gradient[0] = matrix[3].gradient[0] = 1;
  checkPhysical(determinant(matrix, 2), 0, 0, 2);

  // The same expansion agrees away from singularities, including negative determinants.
  matrix[0].value = matrix[3].value = -2;
  matrix[1].value = matrix[2].value = 1;
  checkPhysical(determinant(matrix, 2), 3, -4, 2);
  CHECK_THROWS_AS(determinant(matrix, 3), std::invalid_argument);
}

TEST_CASE("Residual amplitude product uses per-electron gradient cross terms", "[wavefunction][residual]")
{
  SpatialJet a(2, 2), b(2, 3);
  a.gradient[0] = 1;
  b.gradient[0] = 4;
  a.gradient[3] = -2;
  b.gradient[3] = 5;
  a.laplacian[0] = 7;
  b.laplacian[0] = 11;
  auto product = multiply(a, b);
  CHECK(product.value == 6);
  CHECK(product.gradient[0] == 11);
  CHECK(product.laplacian[0] == 51);
  CHECK(product.laplacian[1] == -20);
}


TEST_CASE("Residual matrix adjoints preserve node-safe parameter seeds", "[wavefunction][residual]")
{
  const auto singular = determinantAdjoints({1,1,1,1}, 2);
  CHECK(singular.value == 0);
  CHECK(singular.matrix_adjoints == std::vector<long double>{1,-1,-1,1});
  const auto zero_row = determinantAdjoints({0,0,0,1}, 2);
  CHECK(zero_row.value == 0);
  CHECK(zero_row.matrix_adjoints == std::vector<long double>{1,0,0,0});
  const auto regular = determinantAdjoints({2,3,5,7}, 2);
  const long double scale = std::exp(regular.log_scale);
  CHECK(static_cast<double>(scale*regular.value) == Approx(-1));
  const double expected[] = {7,-5,-3,2};
  for (std::size_t k=0; k<4; ++k)
    CHECK(static_cast<double>(scale*regular.matrix_adjoints[k]) == Approx(expected[k]));
  CHECK_THROWS_AS(determinantAdjoints({1,2,3}, 2), std::invalid_argument);
}
