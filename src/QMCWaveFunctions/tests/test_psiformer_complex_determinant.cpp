//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in the QMCPACK source tree for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
/** @file test_psiformer_complex_determinant.cpp
 * @brief Generated polynomial and analytic checks of CPU complex determinant algebra.
 */
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexDeterminant.h"
#include <array>
#include <iostream>
#include <string>

namespace det = qmcplusplus::psiformer::determinant;
using C = std::complex<double>;
using W = std::complex<long double>;
using Polynomial = std::array<W, 3>;
int failures = 0, checks = 0;

/// Record one assertion without introducing a test-framework dependency.
void check(bool value, const std::string& description)
{
  ++checks;
  if (!value)
  {
    ++failures;
    std::cerr << "FAIL: " << description << '\n';
  }
}

/// Compare complex values with a combined absolute and relative tolerance.
bool near(C a, C b, double tolerance = 3e-11)
{
  return std::abs(a - b) <= tolerance * (1 + std::abs(b));
}
/// Require explicit rejection of undefined or invalid input.
template<class F>
void throws(F function, const std::string& description)
{
  bool caught = false;
  try
  {
    function();
  }
  catch (const std::exception&)
  {
    caught = true;
  }
  check(caught, description);
}

/// Materialize only the small amplitudes used by the independent test oracle.
C value(const det::ComplexLogAmplitude& amplitude)
{
  return amplitude.isZero() ? C{} : amplitude.phase * std::exp(amplitude.log_abs);
}
/// Truncate polynomial multiplication above second order, sufficient for exact VGL.
Polynomial multiply(const Polynomial& a, const Polynomial& b)
{
  Polynomial result{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; i + j < 3; ++j)
      result[i + j] += a[i] * b[j];
  return result;
}

/// Independent tiny determinant oracle: polynomial permutation expansion, no inverse.
Polynomial determinantPolynomial(const C* a, const C* g, const C* l, std::size_t n)
{
  if (!n)
    return {W{1}, W{}, W{}};
  std::vector<std::size_t> permutation(n);
  for (std::size_t i = 0; i < n; ++i)
    permutation[i] = i;
  Polynomial result{};
  do
  {
    Polynomial term{W{1}, W{}, W{}};
    std::size_t inversions = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
      const std::size_t entry = i * n + permutation[i];
      term = multiply(term, {W(a[entry]), W(g[entry]), l ? W(l[entry]) / 2.0L : W{}});
      for (std::size_t j = 0; j < i; ++j)
        inversions += permutation[j] > permutation[i];
    }
    for (int k = 0; k < 3; ++k)
      result[k] += (inversions % 2 ? -term[k] : term[k]);
  } while (std::next_permutation(permutation.begin(), permutation.end()));
  return result;
}

/// All entries, including both spin blocks, depend on every target direction.
void polynomialFixture(std::size_t up, std::size_t down)
{
  const std::size_t elements = up * up + down * down, batch = 2 * elements;
  std::vector<C> matrices(batch), gradients(3 * batch), laplacians(batch), adjoint(batch);
  const C coefficients[2] = {{0.7, 0.2}, {-0.3, 0.4}};
  for (std::size_t c = 0; c < 2; ++c)
    for (std::size_t b = 0; b < 2; ++b)
    {
      const std::size_t n = b ? down : up, offset = c * elements + (b ? up * up : 0);
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
        {
          const std::size_t k = offset + i * n + j;
          matrices[k] = C{i == j ? 1.3 + .4 * c : .11 * (k + 1), .17 * (k + 1)};
          laplacians[k] = C{-.13 * (k + 1), .19};
          for (std::size_t d = 0; d < 3; ++d)
            gradients[d * batch + k] = C{.07 * (k + 1) * (d + 1), -.09 * (d + k + 1)};
        }
    }
  det::ComplexDeterminantWorkspace workspace(2, up, down, 3, 1);
  C gradient[3], lap_log, lap_ratio;
  const auto fingerprint = workspace.storageFingerprint();
  const auto result = workspace.evaluateSpatial(matrices.data(), gradients.data(), 3,
      laplacians.data(), 1, gradient, &lap_log, &lap_ratio, coefficients);
  W reference_value{}, reference_lap{}, squared_gradient{};
  for (std::size_t d = 0; d < 3; ++d)
  {
    Polynomial total{};
    for (std::size_t c = 0; c < 2; ++c)
    {
      const std::size_t offset = c * elements;
      const C* a = matrices.data() + offset;
      const C* g = gradients.data() + d * batch + offset;
      const C* l = d == 0 ? laplacians.data() + offset : nullptr;
      auto term = multiply(determinantPolynomial(a, g, l, up),
          determinantPolynomial(a + up * up, g + up * up, l ? l + up * up : nullptr, down));
      for (int k = 0; k < 3; ++k)
        total[k] += W(coefficients[c]) * term[k];
    }
    reference_value = total[0];
    const W expected_gradient = total[1] / total[0];
    check(near(gradient[d], C(expected_gradient)), "polynomial expansion log gradient");
    squared_gradient += expected_gradient * expected_gradient;
    reference_lap += 2.0L * total[2] / total[0];
  }
  check(near(value(result.amplitude), C(reference_value)), "paired channel determinant expansion");
  check(near(lap_ratio, C(reference_lap)), "ordinary trace-square and spin cross terms");
  check(near(lap_log, C(reference_lap - squared_gradient)), "complex logarithmic Laplacian");
  workspace.fillLogMatrixAdjoints(adjoint.data());
  W contraction{};
  for (std::size_t k = 0; k < batch; ++k)
    contraction += W(adjoint[k]) * W(gradients[k]);
  check(near(C(contraction), gradient[0]), "analytic transpose adjoint, no conjugation");
  check(workspace.storageFingerprint() == fingerprint, "prepared operations do not grow storage");
}

/// Exercise value/derivative admission, cancellation and failed-publication contracts.
void edgeCases()
{
  const C pivoted[4] = {{0,0}, {2,1}, {3,-2}, {4,1}};
  det::ComplexDeterminantWorkspace pivot(1, 2);
  const C exact = -pivoted[1] * pivoted[2];
  check(near(value(pivot.evaluate(pivoted).amplitude), exact), "complex LU pivot phase");
  C adjoint[4];
  pivot.fillLogMatrixAdjoints(adjoint);
  check(near(adjoint[0], pivoted[3] / exact), "inverse transpose diagonal");
  check(near(adjoint[1], -pivoted[2] / exact), "inverse transpose off diagonal");

  det::ComplexDeterminantWorkspace sum(2, 1);
  C terms[2] = {{1,2}, {1,2}}, coefficients[2] = {{1,0}, {-1,0}};
  const auto node = sum.evaluate(terms, coefficients);
  check(node.amplitude.isZero() && node.cancellation_ratio == 0, "exact cancellation remains valid value");
  throws([&] { sum.fillLogMatrixAdjoints(adjoint); }, "node derivatives rejected");
  terms[1] = C{1 + 1e-10, 2};
  const auto cancelled = sum.evaluate(terms, coefficients);
  check(!cancelled.amplitude.isZero() && cancelled.cancellation_ratio < 1e-9,
        "near cancellation diagnosed without declaring node");

  C singular[8] = {{1,0},{0,0},{0,0},{1,0},{1,0},{0,0},{0,0},{0,0}};
  det::ComplexDeterminantWorkspace strict(2, 2, 0, 3, 1);
  const auto singular_result = strict.evaluate(singular);
  check(!singular_result.amplitude.isZero() && singular_result.singular_channels == 1 &&
        !singular_result.all_required_inverses_available, "singular zero-valued channel retains strict admission");
  throws([&] { strict.fillLogMatrixAdjoints(adjoint); }, "singular channel derivative cannot omit cofactors");
  C zero_coefficient[2] = {{1,0},{0,0}}, large_adjoint[8];
  check(strict.evaluate(singular, zero_coefficient).all_required_inverses_available,
        "zero coefficient exempts singular channel");
  strict.fillLogMatrixAdjoints(large_adjoint);
  check(large_adjoint[4] == C{} && large_adjoint[7] == C{}, "zero coefficient seeds are zero");

  C g[24]{}, l[8]{}, out[3] = {C{91},C{92},C{93}}, log{94}, ratio{95};
  throws([&] { strict.evaluateSpatial(singular, g, 3, l, 1, out, &log, &ratio); },
         "compound singular spatial request rejected");
  check(out[0] == C{91} && log == C{94} && ratio == C{95}, "failed request does not publish outputs");
  throws([&] { strict.result(); }, "failed request invalidates preceding state");
  terms[0] = C{std::numeric_limits<double>::quiet_NaN(), 0};
  throws([&] { sum.evaluate(terms); }, "nonfinite matrix rejected");
  throws([&] { sum.result(); }, "nonfinite request invalidates state");

  det::ComplexDeterminantWorkspace empty(1, 0, 0);
  check(value(empty.evaluate(nullptr).amplitude) == C{1}, "two empty spin blocks have determinant one");
  det::ComplexDeterminantWorkspace range(1, 2);
  C huge[4] = {{1e200,1e200},{0,0},{0,0},{1e200,-1e200}};
  check(std::abs(range.evaluate(huge).amplitude.log_abs - (400 * std::log(10.) + std::log(2.))) < 1e-11,
        "log representation survives raw determinant overflow");
  C tiny[4] = {{1e-200,0},{0,0},{0,0},{0,1e-200}};
  check(near(range.evaluate(tiny).amplitude.phase, C{0,1}), "underflow-safe phase");
  // Normal inputs also exercise inverse overflow with release FTZ/DAZ enabled.
  C inverse_overflow[4] = {{1e-300,0},{1e-300,0},{1e-300,0},{1.000000000000001e-300,0}};
  const auto unavailable = range.evaluate(inverse_overflow);
  check(!unavailable.amplitude.isZero() && !unavailable.all_required_inverses_available,
        "inverse overflow preserves value availability");
  check(!range.evaluate(pivoted, nullptr, false).all_required_inverses_available,
        "value-only preparation does not advertise inverse availability");
}

/// exp(i*k*x) has lap(log Psi)=0 and kinetic energy k^2/2 (no conjugation).
void planeWave()
{
  const double k = .73;
  C a{.6,.8}, g[3] = {C{0,k} * a,C{},C{}}, l = -k * k * a;
  C gradient[3], log, ratio;
  det::ComplexDeterminantWorkspace workspace(1, 1, 0, 3, 1);
  workspace.evaluateSpatial(&a, g, 3, &l, 1, gradient, &log, &ratio);
  check(near(gradient[0], C{0,k}) && near(log, C{}), "plane wave exact spatial primitives");
  check(near(-.5 * ratio, C{.5*k*k}), "plane wave positive kinetic energy");
}
/// Run only tiny generated CPU fixtures; no model, Python runtime or external data.
int main()
{
  polynomialFixture(2,0);
  polynomialFixture(1,1);
  polynomialFixture(2,1);
  polynomialFixture(0,2);
  edgeCases();
  planeWave();
  std::cout << checks << " checks, " << failures << " failures\n";
  return failures ? 1 : 0;
}
