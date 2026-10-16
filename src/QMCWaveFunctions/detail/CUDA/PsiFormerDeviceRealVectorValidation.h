//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_REAL_VECTOR_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_REAL_VECTOR_VALIDATION_H
#include "PsiFormerDeviceResources.h"
#include "Platforms/CUDA/QueueCUDA.hpp"
#include "PsiFormerDeviceDerivativeKernels.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceMath.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace qmcplusplus::psiformer::device
{
/// Tiny raw-leaf gate, deliberately not a solver/vector owner or public slot API.
inline void validateDerivativeRealVectors(compute::Queue<PlatformKind::CUDA>& queue)
{
  const auto require = [](bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
  };
  const auto reject = [&](auto operation) {
    bool rejected = false;
    try { operation(); }
    catch (const std::logic_error&) { rejected = true; }
    require(rejected, "Real vector invalid wrapper input reached launch");
  };
  constexpr std::size_t maximum = 4097, capacity = 2 * maximum + 8, pieces = 2;
  DeviceBuffer<double> x(capacity), y(capacity), partials(pieces), result(1);
  DeviceBuffer<DerivativeFiniteStatus> scans(pieces), dot_status(pieces), aggregate(1);
  std::vector<double> source(capacity, 19), expected(capacity, 19), actual(capacity);
  int device = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&device), "real vector metric device");
#else
  checkRuntime(cudaGetDevice(&device), "real vector metric device");
#endif
  auto& metrics = deviceMemoryMetrics(device);
  const auto stream = queue.getNative();
  const auto warm = metrics.snapshot();
  std::size_t cases = 0;

  // Caller finite-admits every logical input/output lane, including a partial tail.
  const auto finite = [&](const double* values, std::size_t count, std::size_t offset, std::size_t stride) {
    const DerivativeFiniteScanLayout layout{count, 4096, pieces, offset, stride};
    checkRuntime(launchDerivativeFiniteScan(stream, layout, values, scans.data()), "real vector finite scan");
    queue.sync();
    DerivativeFiniteStatus host[pieces];
    const auto live = derivativeFiniteStatusCount(layout);
    scans.download(host, live);
    for (std::size_t i = 0; i < live; ++i)
      if (host[i].nonfinite_count) return false;
    return true;
  };
  // Existing deterministic dot/finalize returns only one scalar and aggregate status.
  const auto dot = [&](std::size_t count, const double* left, const double* right,
      std::size_t offset, std::size_t stride, bool regular) {
    const auto before = metrics.snapshot();
    const ResponseDotLayout layout{count, 4096, pieces, offset, stride, offset, stride};
    checkRuntime(launchResponseDotDirection(stream, layout, left, right, partials.data(), dot_status.data(),
        result.data(), aggregate.data()), "real vector dot/finalize");
    queue.sync();
    double value;
    DerivativeFiniteStatus status;
    result.download(&value, 1);
    aggregate.download(&status, 1);
    require((status.nonfinite_count == 0) == regular, "Real vector dot finite-domain mismatch");
    require(metrics.snapshot().downloaded_bytes - before.downloaded_bytes == sizeof(double) + sizeof(status),
        "Real vector dot downloaded partials instead of scalar/status");
    return value;
  };
  for (std::size_t count : {std::size_t(1), std::size_t(4095), std::size_t(4096), maximum})
  {
    std::fill(source.begin(), source.end(), 19);
    std::fill(expected.begin(), expected.end(), 19);
    for (std::size_t i = 0; i < count; ++i)
    {
      source[1 + 2 * i] = (int(i % 17) - 8) / 16.0;
      expected[3 + 2 * i] = (int(i % 13) - 6) / 8.0;
    }
    x.upload(source.data(), source.size());
    y.upload(expected.data(), expected.size());
    const DerivativeVectorLayout layout{count, 1, 2, 3, 2};
    require(finite(x.data(), count, 1, 2) && finite(y.data(), count, 3, 2), "Dyadic vector input nonfinite");
    checkRuntime(launchDerivativeZero(stream, layout, y.data()), "real vector zero");
    checkRuntime(launchDerivativeCopy(stream, layout, x.data(), y.data()), "real vector copy");
    checkRuntime(launchDerivativeScale(stream, layout, -2, y.data()), "real vector scale");
    checkRuntime(launchDerivativeAxpy(stream, layout, 0.5, x.data(), y.data()), "real vector AXPY");
    const DerivativeVectorLayout same{count, 3, 2, 3, 2};
    checkRuntime(launchDerivativeAxpy(stream, same, -0.5, y.data(), y.data()), "real vector exact alias");
    require(finite(y.data(), count, 3, 2), "Dyadic vector output nonfinite");
    double square = 0;
    for (std::size_t i = 0; i < count; ++i)
    {
      expected[3 + 2 * i] = -0.75 * source[1 + 2 * i];
      square += expected[3 + 2 * i] * expected[3 + 2 * i];
    }
    y.download(actual.data(), actual.size());
    require(actual == expected, "Real vector strided arithmetic/tail sentinel mismatch");
    const auto value = dot(count, y.data(), y.data(), 3, 2, true);
    require(value == square && std::sqrt(value) == std::sqrt(square), "Real vector dyadic dot/norm mismatch");
    // Even alpha0 may not hide a nonfinite input from caller admission.
    source[1 + 2 * (count - 1)] = std::numeric_limits<double>::quiet_NaN();
    x.upload(source.data(), source.size());
    require(!finite(x.data(), count, 1, 2), "Real vector finite scan missed the final input lane");
    ++cases;
  }

  // Disjoint addressed intervals inside one allocation and equivalent base+offset alias.
  std::fill(source.begin(), source.end(), 0);
  source[0] = 2; source[1] = -3; source[4] = 1; source[5] = 2;
  x.upload(source.data(), source.size());
  checkRuntime(launchDerivativeAxpy(stream, {2, 0, 1, 4, 1}, 2, x.data(), x.data()), "disjoint same allocation");
  checkRuntime(launchDerivativeAxpy(stream, {1, 0, 1, 1, 2}, -0.5, x.data() + 1, x.data()), "equivalent exact span");
  queue.sync();
  x.download(actual.data(), actual.size());
  require(actual[4] == 5 && actual[5] == -4 && actual[1] == -1.5, "Real vector same-allocation alias arithmetic");
  reject([&] { (void)launchDerivativeAxpy(stream, {3, 0, 1, 1, 1}, 1, x.data(), x.data()); });
  reject([&] { (void)launchDerivativeAxpy(stream, {3, 0, 1, 0, 1}, 1, x.data(), x.data() + 1); });
  reject([&] { (void)launchDerivativeAxpy(stream, {3, 0, 2, 1, 2}, 1, x.data(), x.data()); });
  const auto nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
  const auto size_max = std::numeric_limits<std::size_t>::max();
  for (double scalar : {nan, inf, -inf})
  {
    reject([&] { (void)launchDerivativeScale(stream, {1, 0, 0, 0, 1}, scalar, y.data()); });
    reject([&] { (void)launchDerivativeAxpy(stream, {1, 0, 1, 0, 1}, scalar, x.data(), y.data()); });
  }
  reject([&] { (void)launchDerivativeScale(stream, {}, nan, nullptr); });
  reject([&] { (void)launchDerivativeAxpy(stream, {}, nan, nullptr, nullptr); });
  checkRuntime(launchDerivativeScale(stream, {}, 1, nullptr), "empty scale");
  checkRuntime(launchDerivativeAxpy(stream, {}, 1, nullptr, nullptr), "empty AXPY");
  reject([&] { (void)launchDerivativeScale(stream, {1, 0, 0, 0, 0}, 1, y.data()); });
  reject([&] { (void)launchDerivativeAxpy(stream, {1, 0, 0, 0, 1}, 1, x.data(), y.data()); });
  reject([&] { (void)launchDerivativeScale(stream, {1, 0, 0, 0, 1}, 1, nullptr); });
  reject([&] { (void)launchDerivativeAxpy(stream, {1, 0, 1, 0, 1}, 1, nullptr, y.data()); });
  reject([&] { (void)launchDerivativeScale(stream, {2, 0, 0, size_max, 1}, 1, y.data()); });
  reject([&] { (void)launchDerivativeAxpy(stream, {2, 0, size_max, 0, 1}, 1, x.data(), y.data()); });
  reject([&] { (void)launchDerivativeScale(stream, {1, 0, 0, size_max / sizeof(double), 1}, 1, y.data()); });
  const auto grid_overflow = (std::size_t(std::numeric_limits<unsigned int>::max()) + 1) * 128;
  reject([&] { (void)launchDerivativeScale(stream, {grid_overflow, 0, 0, 0, 1}, 1, y.data()); });
  const auto portable_grid_overflow = (std::size_t(std::numeric_limits<int>::max()) + 1) * 128;
  reject([&] { (void)launchDerivativeScale(stream, {portable_grid_overflow, 0, 0, 0, 1}, 1, y.data()); });

  // Finite inputs do not admit an overflowing product or sum, even when FMA could fit.
  const double huge = std::numeric_limits<double>::max() * 0.75, negative = -huge;
  require(device_math::isFiniteBinary64(std::fma(2.0, huge, negative)), "Overflow-cancellation fixture vacuous");
  x.upload(&huge, 1); y.upload(&negative, 1);
  require(finite(x.data(), 1, 0, 1) && finite(y.data(), 1, 0, 1), "Overflow fixture input not finite");
  checkRuntime(launchDerivativeAxpy(stream, {1, 0, 1, 0, 1}, 2, x.data(), y.data()), "overflowing AXPY product");
  require(!finite(y.data(), 1, 0, 1), "AXPY FMA silently rescued an overflowing product");
  y.upload(&huge, 1);
  checkRuntime(launchDerivativeScale(stream, {1, 0, 0, 0, 1}, 2, y.data()), "overflowing scale");
  require(!finite(y.data(), 1, 0, 1), "Real scale overflow admitted");
  y.upload(&huge, 1);
  checkRuntime(launchDerivativeAxpy(stream, {1, 0, 1, 0, 1}, 1, x.data(), y.data()), "overflowing AXPY sum");
  require(!finite(y.data(), 1, 0, 1), "Real AXPY sum overflow admitted");
  (void)dot(1, x.data(), x.data(), 0, 1, false);
  std::fill(source.begin(), source.end(), 1);
  std::fill(expected.begin(), expected.end(), std::numeric_limits<double>::max() / 2);
  x.upload(source.data(), source.size()); y.upload(expected.data(), expected.size());
  (void)dot(4096, x.data(), y.data(), 0, 1, false);

  // Product/add must also remain separately rounded in the finite cancellation case.
  const double alpha = 1 + std::ldexp(1.0, -27), unit = 1 - std::ldexp(1.0, -27), minus_one = -1;
  require(std::fma(alpha, unit, minus_one) != 0, "Finite FMA fixture vacuous");
  x.upload(&unit, 1); y.upload(&minus_one, 1);
  checkRuntime(launchDerivativeAxpy(stream, {1, 0, 1, 0, 1}, alpha, x.data(), y.data()), "separate real rounding");
  require(finite(y.data(), 1, 0, 1), "Separate-rounding output not finite");
  double rounded;
  y.download(&rounded, 1);
  require(rounded == 0, "Real AXPY contracted a finite product/add");
  const auto after = metrics.snapshot();
  require(after.allocation_calls == warm.allocation_calls && after.free_calls == warm.free_calls,
      "Real vector warmed DEVICE allocation/free");
  std::cout << "PASS checked real vector leaves counts=" << cases
            << " zero_copy_scale_axpy_dot_norm=1 exact_alias=1 partial_overlap_reject=1 separate_rounding=1"
            << " product_sum_dot_overflow_reject=1 dot_download_bytes=" << sizeof(double) + sizeof(DerivativeFiniteStatus)
            << " warm_DEVICE_allocations=0 scope=device_wide raw_boundary=caller_finite_scans\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
