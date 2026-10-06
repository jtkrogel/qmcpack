//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file test_psiformer_determinant.cpp
 * @brief External-data-free tests of stable PsiFormer determinant reductions.
 */

#include "QMCWaveFunctions/PsiFormer/PsiFormerDeterminant.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace det = qmcplusplus::psiformer::determinant;

namespace
{

int checks   = 0;
int failures = 0;

/// Record one assertion while keeping this standalone test independent of Catch2.
void check(bool condition, const std::string& message)
{
  ++checks;
  if (!condition)
  {
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
  }
}

/// Compare finite scalar results with combined relative and absolute tolerances.
bool near(double actual, double expected, double relative = 2.0e-12,
          double absolute = 2.0e-13)
{
  return std::abs(actual - expected) <=
      absolute + relative * std::max(std::abs(actual), std::abs(expected));
}

template<class Callable>
/// Check that an invalid or mathematically undefined operation is rejected.
void checkThrows(Callable&& callable, const std::string& message)
{
  bool threw = false;
  try
  {
    callable();
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  check(threw, message);
}

/// Compute a small determinant by its permutation expansion as an independent oracle.
long double referenceDeterminant(const double* matrix, std::size_t n)
{
  std::vector<std::size_t> permutation(n);
  for (std::size_t i = 0; i < n; ++i)
    permutation[i] = i;
  long double result = 0.0L;
  do
  {
    std::size_t inversions = 0;
    long double product    = 1.0L;
    for (std::size_t row = 0; row < n; ++row)
    {
      product *= static_cast<long double>(matrix[row * n + permutation[row]]);
      for (std::size_t previous = 0; previous < row; ++previous)
        inversions += permutation[previous] > permutation[row] ? 1 : 0;
    }
    result += inversions % 2 == 0 ? product : -product;
  } while (std::next_permutation(permutation.begin(), permutation.end()));
  return result;
}

/// Check row-pivot sign bookkeeping, inverse construction, and pivot diagnostics.
void testPivotedFactorizationAndInverse()
{
  const std::array<double, 4> matrix{0.0, 2.0, 3.0, 4.0};
  det::RealOpenDeterminantWorkspace workspace(1, 2);
  const auto result = workspace.evaluate(matrix.data());
  check(result.amplitude.phase == -1.0, "row pivot preserves determinant sign");
  check(near(result.amplitude.log_abs, std::log(6.0)), "pivoted log determinant");
  check(result.singular_channels == 0, "pivoted matrix is nonsingular");
  check(result.all_required_inverses_available, "pivoted inverse is available");
  check(workspace.permutation(0)[0] == 1 && workspace.permutation(0)[1] == 0,
        "pivot permutation is retained");

  const double* inverse = workspace.inverse(0);
  const std::array<double, 4> expected{-2.0 / 3.0, 1.0 / 3.0, 0.5, 0.0};
  for (std::size_t element = 0; element < expected.size(); ++element)
    check(near(inverse[element], expected[element]), "pivoted inverse element");

  const auto& factor = workspace.factorization(0);
  check(factor.status == det::ChannelFactorizationStatus::REGULAR,
        "regular factorization status");
  check(factor.minimum_scaled_pivot > 0.0, "minimum pivot diagnostic is populated");
}

/// Check exact singular contributions and the explicit exact-node representation.
void testSingularChannelsAndExactNode()
{
  const std::array<double, 8> matrices{
      1.0, 2.0, 2.0, 4.0,
      1.0, 0.0, 0.0, 1.0};
  det::RealOpenDeterminantWorkspace workspace(2, 2);
  const auto result = workspace.evaluate(matrices.data());
  check(result.singular_channels == 1, "one singular channel is recorded");
  check(result.amplitude.phase == 1.0 && near(result.amplitude.log_abs, 0.0),
        "singular channel contributes an exact zero to a valid sum");
  check(!result.all_required_inverses_available,
        "singular derivative channel is reported rather than silently discarded");
  check(workspace.inverse(0) == nullptr, "singular channel exposes no inverse");
  std::array<double, 8> adjoints{};
  checkThrows([&] { workspace.fillLogAmplitudeMatrixAdjoints(adjoints.data()); },
              "reverse seed rejects an exact singular channel");

  const std::array<double, 8> all_singular{
      1.0, 2.0, 2.0, 4.0,
      0.0, 0.0, 0.0, 0.0};
  const auto node = workspace.evaluate(all_singular.data());
  check(node.amplitude.isZero(), "all-singular channel sum is an exact node");
  check(node.amplitude.log_abs == -std::numeric_limits<double>::infinity(),
        "exact node has negative-infinite log magnitude");
  check(det::realValue(node.amplitude) == 0.0, "exact node materializes as zero");
}

/// Check that log amplitudes survive raw determinant over- and underflow.
void testOverflowAndUnderflow()
{
  const std::array<double, 4> huge{1.0e200, 0.0, 0.0, -2.0e200};
  det::RealOpenDeterminantWorkspace workspace(1, 2);
  auto result = workspace.evaluate(huge.data());
  check(result.amplitude.phase == -1.0, "overflowing raw determinant sign");
  check(near(result.amplitude.log_abs, std::log(2.0) + 400.0 * std::log(10.0),
             2.0e-14, 2.0e-12),
        "overflowing raw determinant retains finite log magnitude");
  check(!det::isFiniteReal(det::realValue(result.amplitude)),
        "raw value overflow is deferred to explicit materialization");

  const std::array<double, 4> tiny{1.0e-200, 0.0, 0.0, 2.0e-200};
  result = workspace.evaluate(tiny.data());
  check(result.amplitude.phase == 1.0, "underflowing raw determinant sign");
  check(near(result.amplitude.log_abs, std::log(2.0) - 400.0 * std::log(10.0),
             2.0e-14, 2.0e-12),
        "underflowing raw determinant retains finite log magnitude");
  check(det::realValue(result.amplitude) == 0.0,
        "raw value underflow is deferred to explicit materialization");
  check(workspace.inverse(0) && near(workspace.inverse(0)[0], 1.0e200, 2.0e-14, 0.0),
        "scaled factorization retains tiny-matrix inverse");
}

/// Exercise near-singular matrices and discontinuous LU pivot selection boundaries.
void testNearlySingularAndPivotChanges()
{
  det::RealOpenDeterminantWorkspace workspace(1, 2);
  for (int exponent : {20, 35, 48})
  {
    const double delta = std::ldexp(1.0, -exponent);
    const std::array<double, 4> near_singular{
        1.0, 1.0, 1.0, 1.0 + delta};
    const auto result = workspace.evaluate(near_singular.data());
    const long double reference =
        referenceDeterminant(near_singular.data(), 2);
    check(result.amplitude.phase == 1.0,
          "near-singular ladder determinant sign");
    check(near(result.amplitude.log_abs,
               std::log(static_cast<double>(reference)), 2.0e-10, 2.0e-10),
          "near-singular ladder remains evaluable");
    check(workspace.factorization(0).minimum_scaled_pivot <= 2.0 * delta,
          "near-singular ladder is visible through pivot diagnostics");
  }

  for (double epsilon : {-1.0e-12, 0.0, 1.0e-12})
  {
    const std::array<double, 4> pivot_change{1.0 + epsilon, 2.0,
                                             1.0 - epsilon, 3.0};
    const auto result = workspace.evaluate(pivot_change.data());
    const long double determinant = referenceDeterminant(pivot_change.data(), 2);
    check(result.amplitude.phase == (determinant > 0.0L ? 1.0 : -1.0),
          "pivot-change sign parity");
    check(near(result.amplitude.log_abs,
               std::log(std::abs(static_cast<double>(determinant))), 5.0e-13, 5.0e-13),
          "pivot-change log parity");
  }
}

/// Check compensated signed cancellation and the divergent gradient near a node.
void testSignedCancellationAndNearNodeGradient()
{
  const std::array<double, 8> matrices{
      1.0, 0.0, 0.0, 1.0,
      1.0, 0.0, 0.0, 1.0};
  const double delta = std::ldexp(1.0, -40);
  const std::array<double, 2> coefficients{1.0, -(1.0 - delta)};
  det::RealOpenDeterminantWorkspace workspace(2, 2, 1, 0);
  const auto result = workspace.evaluate(matrices.data(), coefficients.data());
  check(result.amplitude.phase == 1.0, "opposite-sign cancellation phase");
  check(near(result.amplitude.log_abs, std::log(delta), 3.0e-5, 3.0e-5),
        "opposite-sign cancellation log magnitude");
  check(std::abs(workspace.channelWeight(0)) > 1.0e11L,
        "near-node normalized channel weight is retained in extended precision");

  std::array<double, 8> gradient{};
  gradient[0] = 1.0; // d A_0(0,0) / dx
  double output_gradient = 0.0;
  workspace.evaluateSpatial(matrices.data(), gradient.data(), 1, nullptr, 0,
                            &output_gradient, nullptr, nullptr, coefficients.data());
  check(near(output_gradient, 1.0 / delta, 4.0e-5, 2.0),
        "near-node analytic log gradient");

  const std::array<double, 2> cancelling_coefficients{1.0, -1.0};
  const auto node = workspace.evaluate(matrices.data(), cancelling_coefficients.data());
  check(node.amplitude.isZero(), "exact opposite-sign cancellation is a node");
  checkThrows(
      [&] {
        workspace.evaluateSpatial(matrices.data(), gradient.data(), 1, nullptr, 0,
                                  &output_gradient, nullptr, nullptr,
                                  cancelling_coefficients.data());
      },
      "log gradient is explicitly undefined at cancellation node");
}

/// Preserve a residual channel whose determinant lies below binary64 exp range.
void testWideGapSignedCancellation()
{
  const double tiny = std::exp(-400.0);
  const std::array<double, 12> matrices{
      1.0, 0.0, 0.0, 1.0,
      1.0, 0.0, 0.0, 1.0,
      tiny, 0.0, 0.0, tiny};
  const std::array<double, 3> coefficients{1.0, -1.0, 1.0};
  det::RealOpenDeterminantWorkspace workspace(3, 2);

  const auto result = workspace.evaluateValue(matrices.data(), coefficients.data());
  check(result.amplitude.phase == 1.0,
        "wide-gap cancellation preserves residual phase");
  check(near(result.amplitude.log_abs, -800.0, 2.0e-14, 2.0e-12),
        "wide-gap cancellation preserves residual log magnitude");
}

/// A failed evaluation must invalidate rather than expose the preceding result.
void testFailedEvaluationInvalidatesResult()
{
  const std::array<double, 4> identity{1.0, 0.0, 0.0, 1.0};
  det::RealOpenDeterminantWorkspace workspace(1, 2);
  workspace.evaluate(identity.data());
  check(workspace.result().amplitude.phase == 1.0,
        "successful determinant result is initially observable");

  auto invalid = identity;
  invalid[3]   = std::numeric_limits<double>::quiet_NaN();
  checkThrows([&] { workspace.evaluate(invalid.data()); },
              "non-finite determinant input is rejected");
  checkThrows([&] { (void)workspace.result(); },
              "failed determinant evaluation invalidates prior result");

  invalid[3] = std::numeric_limits<double>::infinity();
  checkThrows([&] { workspace.evaluate(invalid.data()); },
              "infinite determinant input is rejected");
  const double invalid_coefficient =
      std::numeric_limits<double>::infinity();
  checkThrows(
      [&] { workspace.evaluate(identity.data(), &invalid_coefficient); },
      "infinite determinant coefficient is rejected");

  const auto recovered = workspace.evaluate(identity.data());
  check(recovered.amplitude.phase == 1.0 && near(recovered.amplitude.log_abs, 0.0),
        "determinant workspace recovers after invalid input");
}

/// Derivative failures must not publish a valid prefix of caller destinations.
void testDerivativePublicationIsAtomic()
{
  const std::array<double, 4> matrix{1.0, 0.0, 0.0, 1.0};
  std::array<double, 8> gradients{};
  gradients[0] = 1.0;
  gradients[4] = std::numeric_limits<double>::quiet_NaN();
  det::RealOpenDeterminantWorkspace workspace(1, 2, 2, 0);
  std::array<double, 2> outputs{17.0, -23.0};

  checkThrows(
      [&] {
        workspace.evaluateSpatial(matrix.data(), gradients.data(), 2, nullptr, 0,
                                  outputs.data(), nullptr, nullptr);
      },
      "non-finite determinant gradient lane is rejected");
  check(outputs == std::array<double, 2>{17.0, -23.0},
        "failed spatial derivative leaves every destination unchanged");
  checkThrows([&] { (void)workspace.result(); },
              "failed spatial derivative invalidates compound result");

  gradients[4] = 2.0;
  workspace.evaluateSpatial(matrix.data(), gradients.data(), 2, nullptr, 0,
                            outputs.data(), nullptr, nullptr);
  check(near(outputs[0], 1.0) && near(outputs[1], 2.0),
        "spatial derivative workspace recovers after invalid input");

  const std::array<double, 4> exact_node{};
  std::array<double, 4> node_gradient{};
  double node_output = 41.0;
  det::RealOpenDeterminantWorkspace node_workspace(1, 2, 1, 0);
  checkThrows(
      [&] {
        node_workspace.evaluateSpatial(
            exact_node.data(), node_gradient.data(), 1, nullptr, 0,
            &node_output, nullptr, nullptr);
      },
      "exact-node spatial derivative is rejected");
  check(node_output == 41.0,
        "exact-node spatial failure leaves its destination unchanged");
  checkThrows([&] { (void)node_workspace.result(); },
              "exact-node spatial failure invalidates compound result");

  const std::array<double, 8> singular_channel{
      1.0, 2.0, 2.0, 4.0,
      1.0, 0.0, 0.0, 1.0};
  std::array<double, 8> singular_gradient{};
  double singular_output = -43.0;
  det::RealOpenDeterminantWorkspace singular_workspace(2, 2, 1, 0);
  checkThrows(
      [&] {
        singular_workspace.evaluateSpatial(
            singular_channel.data(), singular_gradient.data(), 1, nullptr, 0,
            &singular_output, nullptr, nullptr);
      },
      "spatial derivative requiring a singular-channel inverse is rejected");
  check(singular_output == -43.0,
        "singular-channel spatial failure leaves its destination unchanged");
  checkThrows([&] { (void)singular_workspace.result(); },
              "singular-channel spatial failure invalidates compound result");

  std::array<double, 12> lap_gradients{};
  std::array<double, 4> laplacians{};
  laplacians[0] = std::numeric_limits<double>::infinity();
  det::RealOpenDeterminantWorkspace lap_workspace(1, 2, 3, 1);
  std::array<double, 3> lap_output_gradient{5.0, 6.0, 7.0};
  double lap_output_log   = 8.0;
  double lap_output_ratio = 9.0;
  checkThrows(
      [&] {
        lap_workspace.evaluateSpatial(
            matrix.data(), lap_gradients.data(), 3, laplacians.data(), 1,
            lap_output_gradient.data(), &lap_output_log, &lap_output_ratio);
      },
      "non-finite determinant Laplacian lane is rejected");
  check(lap_output_gradient == std::array<double, 3>{5.0, 6.0, 7.0} &&
            lap_output_log == 8.0 && lap_output_ratio == 9.0,
        "failed Laplacian derivative leaves every destination unchanged");

  const std::array<double, 8> near_cancelling_channels{
      1.0, 0.0, 0.0, 1.0,
      1.0e-300, 0.0, 0.0, 1.0};
  const double residual = std::ldexp(1.0, -48);
  const std::array<double, 2> coefficients{
      1.0, -1.0e300 * (1.0 - residual)};
  det::RealOpenDeterminantWorkspace reverse_workspace(2, 2);
  reverse_workspace.evaluate(near_cancelling_channels.data(),
                             coefficients.data());
  std::array<double, 8> adjoints;
  adjoints.fill(31.0);
  checkThrows(
      [&] { reverse_workspace.fillLogAmplitudeMatrixAdjoints(adjoints.data()); },
      "overflowing reverse determinant lane is rejected");
  check(std::all_of(adjoints.begin(), adjoints.end(),
                    [](double value) { return value == 31.0; }),
        "failed reverse derivative leaves every destination unchanged");
}

/// Construct a two-channel matrix family and its first/trace-second derivatives.
void fillSpatialMatrices(double x,
                         std::array<double, 8>& matrices,
                         std::array<double, 24>& gradients,
                         std::array<double, 8>& laplacians)
{
  matrices = {1.0 + x, 0.2, 0.1, 0.8 - x,
              0.5 - 0.3 * x, 0.4, 0.2, -0.7 + 0.1 * x};
  gradients.fill(0.0);
  gradients[0] = 1.0;
  gradients[3] = -1.0;
  gradients[4] = -0.3;
  gradients[7] = 0.1;
  laplacians.fill(0.0); // Matrix entries are linear; determinant curvature is analytic.
}

/// Evaluate the reference log amplitude used by coordinate finite differences.
double spatialLogAmplitude(double x)
{
  std::array<double, 8> matrices{};
  std::array<double, 24> gradients{};
  std::array<double, 8> laplacians{};
  fillSpatialMatrices(x, matrices, gradients, laplacians);
  det::RealOpenDeterminantWorkspace workspace(2, 2);
  return workspace.evaluate(matrices.data()).amplitude.log_abs;
}

/// Validate analytic determinant gradient and Laplacian primitives by finite differences.
void testSpatialFiniteDifferences()
{
  constexpr double x = 0.1;
  std::array<double, 8> matrices{};
  std::array<double, 24> gradients{};
  std::array<double, 8> laplacians{};
  fillSpatialMatrices(x, matrices, gradients, laplacians);

  det::RealOpenDeterminantWorkspace workspace(2, 2, 3, 1);
  std::array<double, 3> output_gradient{};
  std::array<double, 1> output_lap_log{};
  std::array<double, 1> output_lap_ratio{};
  const auto result = workspace.evaluateSpatial(
      matrices.data(), gradients.data(), 3, laplacians.data(), 1,
      output_gradient.data(), output_lap_log.data(), output_lap_ratio.data());
  check(result.amplitude.phase == 1.0, "spatial test amplitude sign");

  const double psi        = 0.35 + 0.06 * x - 1.03 * x * x;
  const double psi_first  = 0.06 - 2.06 * x;
  const double psi_second = -2.06;
  const double expected_gradient = psi_first / psi;
  const double expected_ratio    = psi_second / psi;
  const double expected_log_lap  = expected_ratio - expected_gradient * expected_gradient;
  check(near(output_gradient[0], expected_gradient, 5.0e-13, 5.0e-13),
        "analytic determinant gradient primitive");
  check(output_gradient[1] == 0.0 && output_gradient[2] == 0.0,
        "unused Cartesian lanes remain zero");
  check(near(output_lap_ratio[0], expected_ratio, 5.0e-13, 5.0e-13),
        "analytic determinant Laplacian ratio primitive");
  check(near(output_lap_log[0], expected_log_lap, 5.0e-13, 5.0e-13),
        "analytic determinant logarithmic Laplacian primitive");

  const double gradient_step = 2.0e-6;
  const double finite_gradient =
      (spatialLogAmplitude(x + gradient_step) - spatialLogAmplitude(x - gradient_step)) /
      (2.0 * gradient_step);
  check(near(output_gradient[0], finite_gradient, 2.0e-9, 2.0e-9),
        "coordinate finite-difference gradient parity");

  const double lap_step = 2.0e-4;
  const double finite_lap =
      (spatialLogAmplitude(x + lap_step) - 2.0 * spatialLogAmplitude(x) +
       spatialLogAmplitude(x - lap_step)) /
      (lap_step * lap_step);
  check(near(output_lap_log[0], finite_lap, 2.0e-7, 2.0e-7),
        "coordinate finite-difference second-derivative parity");
}

/// Validate reverse matrix seeds against independent elementwise finite differences.
void testReverseSeedFiniteDifferences()
{
  std::array<double, 8> matrices{
      1.2, 0.1, -0.2, 0.9,
      0.6, -0.3, 0.4, 1.1};
  const std::array<double, 2> coefficients{1.0, -0.35};
  det::RealOpenDeterminantWorkspace workspace(2, 2);
  workspace.evaluate(matrices.data(), coefficients.data());
  std::array<double, 8> adjoints{};
  workspace.fillLogAmplitudeMatrixAdjoints(adjoints.data());

  constexpr double step = 2.0e-6;
  for (std::size_t element = 0; element < matrices.size(); ++element)
  {
    const double original = matrices[element];
    matrices[element]     = original + step;
    const double plus = workspace.evaluate(matrices.data(), coefficients.data()).amplitude.log_abs;
    matrices[element] = original - step;
    const double minus = workspace.evaluate(matrices.data(), coefficients.data()).amplitude.log_abs;
    matrices[element] = original;
    const double finite_difference = (plus - minus) / (2.0 * step);
    check(near(adjoints[element], finite_difference, 2.0e-9, 2.0e-9),
          "matrix reverse seed finite-difference parity");
  }
}

/// Check larger matrices against a permutation oracle and audit allocation stability.
void testFourByFourReferenceAndStorageStability()
{
  const std::array<double, 32> matrices{
      1.0, 0.2, -0.1, 0.4,
      0.3, 1.2, 0.5, -0.2,
      -0.4, 0.1, 0.9, 0.6,
      0.2, -0.3, 0.7, 1.1,
      0.8, -0.6, 0.2, 0.1,
      -0.3, 1.4, 0.5, 0.2,
      0.7, 0.1, -0.9, 0.4,
      0.2, 0.3, 0.6, 1.3};
  det::RealOpenDeterminantWorkspace workspace(2, 4, 3, 1);
  const std::size_t fingerprint = workspace.storageFingerprint();
  const auto result = workspace.evaluate(matrices.data());
  long double reference_sum = referenceDeterminant(matrices.data(), 4) +
      referenceDeterminant(matrices.data() + 16, 4);
  check(result.amplitude.phase == (reference_sum > 0.0L ? 1.0 : -1.0),
        "four-by-four reference sign parity");
  check(near(result.amplitude.log_abs,
             std::log(std::abs(static_cast<double>(reference_sum))), 2.0e-12, 2.0e-12),
        "four-by-four reference log parity");
  for (int repetition = 0; repetition < 20; ++repetition)
    workspace.evaluate(matrices.data());
  check(workspace.storageFingerprint() == fingerprint,
        "warm determinant calls preserve all backing storage");
  check(workspace.storageBytes() > 0, "workspace reports determinant storage");
}

} // namespace

/// Run every standalone determinant test and report an aggregate result.
int main()
{
  testPivotedFactorizationAndInverse();
  testSingularChannelsAndExactNode();
  testOverflowAndUnderflow();
  testNearlySingularAndPivotChanges();
  testSignedCancellationAndNearNodeGradient();
  testWideGapSignedCancellation();
  testFailedEvaluationInvalidatesResult();
  testDerivativePublicationIsAtomic();
  testSpatialFiniteDifferences();
  testReverseSeedFiniteDifferences();
  testFourByFourReferenceAndStorageStability();

  if (failures != 0)
  {
    std::cerr << failures << " failures across " << checks << " checks\n";
    return EXIT_FAILURE;
  }
  std::cout << "PASS: " << checks
            << " stable determinant checks (value, spatial, reverse, adversarial)\n";
  return EXIT_SUCCESS;
}
