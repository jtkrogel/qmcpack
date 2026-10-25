//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//////////////////////////////////////////////////////////////////////////////////////
/** @file test_KFACDenseSolver.cpp
 * @brief Independent compact vendor factor/solve and canonical-restore oracles.
 */
#include "QMCDrivers/WFTrain/KFACPreconditioner.h"
#include "QMCDrivers/WFTrain/KFACDenseSolver.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>
#if defined(ENABLE_CUDA)
#include "QMCWaveFunctions/detail/CUDA/PsiFormerDeviceResources.h"
#endif
namespace qmcplusplus::wftrain
{
TEST_CASE("Vendor KFAC factors preserve bias repeats and canonical restores",
          "[.][drivers][wftrain][kfac_device]")
{
  const StructuredParameterSchema schema{"kfac/vendor-test",
      {{"affine/w", {2, 2}, 0, 4, ParameterScalarDomain::REAL64, true, "network"},
       {"affine/b", {2}, 4, 2, ParameterScalarDomain::REAL64, true, "network"},
       {"fallback", {2}, 6, 2, ParameterScalarDomain::REAL64, true, "network"},
       {"frozen", {1}, 8, 1, ParameterScalarDomain::REAL64, false, "network"}}};
  const KFACBlockRegistry registry{schema, {{"dense", 0, 1, 2, 2}}, {2}};
  KFACFactorAccumulator accumulator(registry, 3);
  const std::array<DerivativeReal, 4> activations{1, 0, 0, 2}, sensitivities{2, 1, 1, 3};
  const std::array<DerivativeReal, 9> score{0, 0, 0, 0, 0, 0, 3, 4, 10};
  const std::array<DerivativeReal, 9> phase{0, 0, 0, 0, 0, 0, 4, -3, 0};
  accumulator.beginSample(1);
  accumulator.addObservation({0, {activations.data(), activations.size()},
      {sensitivities.data(), sensitivities.size()}, 2});
  accumulator.addFallbackScores({score.data(), score.size()}, {phase.data(), phase.size()});
  accumulator.endSample();
  reduceKFACFactorStatistics(accumulator);
  const KFACPreconditioner host(accumulator, {0.04, 0.5});
  const std::vector<DerivativeValue> gradient{1, 2, 3, 4, 5, 6, 7, 8, 0};
  std::vector<DerivativeValue> expected(gradient.size()), actual(gradient.size()), restored(gradient.size());
  const StructuredParameterVectorConstView view(accumulator.registry().parameterSchema(), 3,
      {gradient.data(), gradient.size()});
  host.apply(view, {expected.data(), expected.size()});
  // Independent full Kronecker solve, with BOTH factors non-diagonal. This
  // checks orientation, augmented bias placement and repeat scaling together.
  const long double split = std::sqrt(.08L);
  const std::array<long double, 9> A{1 + split, 0, 1, 0, 4 + split, 2, 1, 2, 2 + split};
  const std::array<long double, 4> G{5 + split, 5, 5, 10 + split};
  std::array<long double, 36> matrix{};
  std::array<long double, 6> independent{1, 2, 3, 4, 5, 6};
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      for (std::size_t o = 0; o < 2; ++o)
        for (std::size_t p = 0; p < 2; ++p)
          matrix[(2 * i + o) * 6 + 2 * j + p] = A[i * 3 + j] * G[o * 2 + p] / 2;
  for (std::size_t k = 0; k < 6; ++k)
  {
    std::size_t pivot = k;
    for (std::size_t row = k + 1; row < 6; ++row)
      if (std::abs(matrix[row * 6 + k]) > std::abs(matrix[pivot * 6 + k])) pivot = row;
    for (std::size_t col = 0; col < 6; ++col) std::swap(matrix[k * 6 + col], matrix[pivot * 6 + col]);
    std::swap(independent[k], independent[pivot]);
    const auto diagonal = matrix[k * 6 + k];
    REQUIRE(std::abs(diagonal) > 0);
    for (std::size_t col = k; col < 6; ++col) matrix[k * 6 + col] /= diagonal;
    independent[k] /= diagonal;
    for (std::size_t row = 0; row < 6; ++row)
      if (row != k)
      {
        const auto scale = matrix[row * 6 + k];
        for (std::size_t col = k; col < 6; ++col) matrix[row * 6 + col] -= scale * matrix[k * 6 + col];
        independent[row] -= scale * independent[k];
      }
  }
  for (std::size_t i = 0; i < independent.size(); ++i)
    CHECK(std::abs(expected[i].real() - independent[i]) <= 2e-12L + 2e-12L * std::abs(independent[i]));
#if defined(ENABLE_CUDA)
  for (const int route : {0, 1, 2})
  {
    const auto mode = route == 1 ? KFACDenseCapacity::STREAMED : KFACDenseCapacity::RESIDENT;
    const bool authority = route == 2;
    INFO("capacity=" << static_cast<int>(mode));
    INFO("resident_authority=" << authority);
    auto resource = KFACDenseSolver::prepare(accumulator.registry(), 1024 * 1024, 1024 * 1024, mode, authority);
    CHECK(resource->capacity() == mode);
    int selected_device = -1;
#ifdef QMC_CUDA2HIP
    psiformer::device::checkRuntime(hipGetDevice(&selected_device), "KFAC fixture selected device");
#else
    psiformer::device::checkRuntime(cudaGetDevice(&selected_device), "KFAC fixture selected device");
#endif
    auto& metrics = psiformer::device::deviceMemoryMetrics(selected_device);
    const auto before_prepare = metrics.snapshot();
    const KFACPreconditioner device(accumulator, {0.04, 0.5}, resource);
    if (authority)
    {
      CHECK(device.denseGeneration()->complete());
      CHECK(device.denseGeneration()->materializedBytes() == 0);
      CHECK(device.inverseSnapshot().affineNumericBytes() == 0);
      CHECK(metrics.snapshot().downloaded_bytes - before_prepare.downloaded_bytes ==
          2 * sizeof(int) + 2 * sizeof(std::size_t) + 5 * sizeof(double));
    }
    device.apply(view, {actual.data(), actual.size()});
    auto imaginary_tangent = gradient;
    imaginary_tangent[6] = {7, .3};
    CHECK_THROWS_AS(device.apply({schema, 3, {imaginary_tangent.data(), imaginary_tangent.size()}},
        {restored.data(), restored.size()}), std::invalid_argument);
    if (authority)
    {
      const auto before_rebind = metrics.snapshot();
      const KFACPreconditioner shared(registry, 3, device.inverseSnapshot(), resource, device.denseGeneration());
      shared.apply(view, {restored.data(), restored.size()});
      CHECK(restored == actual);
      CHECK(device.denseGeneration()->materializedBytes() == 0);
      // The direction is still explicitly HOST-bound; only RHS result D2H is expected.
      CHECK(metrics.snapshot().downloaded_bytes - before_rebind.downloaded_bytes == 6 * sizeof(double));
      auto different = resource->beginGeneration();
      CHECK_THROWS_AS(KFACPreconditioner(registry, 3, device.inverseSnapshot(), resource, different), std::invalid_argument);
      auto other_resource = KFACDenseSolver::prepare(registry, 1024 * 1024, 1024 * 1024,
          KFACDenseCapacity::RESIDENT, true);
      CHECK_THROWS_AS(KFACPreconditioner(registry, 3, device.inverseSnapshot(), other_resource), std::invalid_argument);
    }
    const auto before_export = metrics.snapshot();
    const auto canonical = device.exportInverseState();
    if (authority)
    {
      CHECK(metrics.snapshot().downloaded_bytes - before_export.downloaded_bytes == 13 * sizeof(double));
      CHECK(device.denseGeneration()->materializedBytes() == 13 * sizeof(double));
      const auto before_repeat = metrics.snapshot();
      CHECK(device.exportInverseState().values == canonical.values);
      CHECK(metrics.snapshot().downloaded_bytes == before_repeat.downloaded_bytes);
      std::vector<double> supplied_a(canonical.values.begin(), canonical.values.begin() + 9);
      std::vector<double> supplied_g(canonical.values.begin() + 9, canonical.values.begin() + 13);
      std::array<double, 6> matrix{1, 2, 3, 4, 5, 6};
      device.denseGeneration()->apply(0, supplied_a, supplied_g, matrix.data(), matrix.size());
      supplied_a[0] += .01;
      CHECK_THROWS_AS(device.denseGeneration()->apply(0, supplied_a, supplied_g, matrix.data(), matrix.size()), std::invalid_argument);
      supplied_a[0] -= .01;
      supplied_g.insert(supplied_g.begin(), supplied_a.back());
      supplied_a.pop_back();
      CHECK_THROWS_AS(device.denseGeneration()->apply(0, supplied_a, supplied_g, matrix.data(), matrix.size()), std::invalid_argument);
      {
        auto mixed = resource->beginGeneration();
        std::vector<double> a{1, 0, 0, 0, 1, 0, 0, 0, 1}, g{1, 0, 0, 1};
        mixed->factorize(0, a, g);
        CHECK_THROWS_WITH(mixed->import(0, a, g), "KFAC generation cannot mix resident factors and imported HOST authority");
      }
      CHECK_THROWS_AS(KFACDenseSolver::prepare(registry, 1024 * 1024, 1024 * 1024,
          KFACDenseCapacity::STREAMED, true), std::length_error);
      CHECK_THROWS_AS(KFACDenseSolver::prepare(registry, 1024 * 1024,
          resource->hostBytes() + 2 * 13 * sizeof(double) - 1,
          KFACDenseCapacity::RESIDENT, true), std::length_error);
      const KFACPreconditioner explicit_host(registry, 3, device.inverseSnapshot());
      explicit_host.apply(view, {restored.data(), restored.size()});
      for (std::size_t i = 0; i < actual.size(); ++i)
        CHECK(std::abs(restored[i] - expected[i]) <= 2e-12 + 2e-12 * std::abs(expected[i]));
    }
    {
      // Import the SAME canonical lower factors, never refactor on restart.
      const KFACPreconditioner loaded(accumulator.registry(), 3, canonical, resource);
      CHECK(loaded.exportInverseState().values == canonical.values);
      const auto before_loaded = metrics.snapshot();
      loaded.apply(view, {restored.data(), restored.size()});
      CHECK(restored == actual);
      if (authority)
      {
        CHECK_FALSE(loaded.denseGeneration()->deviceAuthority());
        CHECK(loaded.denseGeneration()->materializedBytes() == 0);
        CHECK(metrics.snapshot().downloaded_bytes - before_loaded.downloaded_bytes == 6 * sizeof(double));
      }
      const KFACPreconditioner reused(accumulator.registry(), 3, canonical, resource, device.denseGeneration());
      reused.apply(view, {restored.data(), restored.size()});
      CHECK(restored == actual);
      auto changed = canonical;
      changed.values[0] += .01;
      CHECK_THROWS_AS(KFACPreconditioner(accumulator.registry(), 3, changed, resource, device.denseGeneration()),
          std::invalid_argument);
    }
    double maximum = 0;
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
      maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
      CHECK(std::abs(actual[i] - expected[i]) <= 2e-12 + 2e-12 * std::abs(expected[i]));
      if (i < independent.size())
        CHECK(std::abs(actual[i].real() - independent[i]) <= 2e-12L + 2e-12L * std::abs(independent[i]));
    }
    double factor_error = 0;
    const auto reference = host.exportInverseState();
    for (std::size_t i = 0; i < reference.values.size(); ++i)
      factor_error = std::max(factor_error, std::abs(reference.values[i] - canonical.values[i]));
    std::ostringstream diagnostic;
    diagnostic.precision(17);
    diagnostic << "KFAC vendor dense oracle capacity=" << static_cast<int>(mode)
               << " max_direction_abs=" << maximum << " max_factor_abs=" << factor_error
               << " canonical_restore_exact=1\n";
    std::clog << diagnostic.str();
    CHECK_THROWS_AS(KFACDenseSolver::prepare(accumulator.registry(), resource->deviceBytes() - 1,
        1024 * 1024, mode, authority), std::length_error);
    {
      auto rejected = resource->beginGeneration();
      std::vector<double> a{-1, 0, 0, 0, 1, 0, 0, 0, 1}, g{1, 0, 0, 1};
      CHECK_THROWS_AS(rejected->factorize(0, a, g), MatrixFreeNumericalError);
      CHECK_THROWS(rejected->finish());
    }
    device.apply(view, {restored.data(), restored.size()});
    CHECK(restored == actual); // Failed pending generation cannot damage accepted factors.
    if (authority)
      for (const auto invalid : {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), 0.})
      {
        auto rejected = resource->beginGeneration();
        std::vector<double> a{invalid, 0, 0, 0, 1, 0, 0, 0, 1}, g{1, 0, 0, 1};
        CHECK_THROWS_AS(rejected->factorize(0, a, g), MatrixFreeNumericalError);
        CHECK_THROWS(rejected->finish());
      }
  }
#else
  CHECK_THROWS_AS(KFACDenseSolver::prepare(registry, 1024 * 1024, 1024 * 1024), std::invalid_argument);
#endif
}
}
