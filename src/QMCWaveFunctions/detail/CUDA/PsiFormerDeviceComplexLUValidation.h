//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_LU_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_LU_VALIDATION_H
#include "PsiFormerDeviceComplexValue.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexDeterminant.h"
#include <chrono>
#include <iostream>
#include <cstring>
#include <cstdlib>

/// Independent determinant oracles; growth is separate to avoid repeating a large stress case in timing trials.
void validateComplexLU(bool growth_only = false)
{
  using namespace qmcplusplus::psiformer;
  using namespace qmcplusplus::psiformer::device;
  using C = std::complex<double>;
  qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA> queue;
  double max_error = 0;
  std::size_t comparisons = 0;
  std::uint64_t fingerprint = 1469598103934665603ull;
  const auto compare = [&](std::size_t n, const std::vector<C>& input,
                           bool expected_failure = false, bool inverse = false) {
    const auto channels = input.size() / (n * n);
    std::vector<cuDoubleComplex> packed(input.size());
    std::vector<cuDoubleComplex> lu(input.size());
    std::vector<cuDoubleComplex> inv(inverse ? input.size() : 0);
    for (std::size_t i = 0; i < input.size(); ++i)
      packed[i] = make_cuDoubleComplex(input[i].real(), input[i].imag());
    DeviceBuffer<cuDoubleComplex> matrix(packed.size());
    DeviceBuffer<cuDoubleComplex> inverse_buffer(inv.size());
    DeviceBuffer<cuDoubleComplex> solve_buffer(inverse ? channels * n : 0);
    DeviceBuffer<std::size_t> permutation(inverse ? channels * n : 0);
    DeviceBuffer<ComplexValueChannel> records(channels);
    DeviceBuffer<std::uint64_t> status(1);
    const auto clear = std::numeric_limits<std::uint64_t>::max();
    matrix.upload(packed.data(), packed.size());
    status.upload(&clear, 1);
    checkRuntime(launchComplexValueFactor(queue.getNative(), channels, n, matrix.data(), records.data(), status.data(),
        inverse_buffer.data(), permutation.data(), solve_buffer.data()), "compact LU");
    queue.sync();
    std::uint64_t code;
    status.download(&code, 1);
    std::vector<ComplexValueChannel> actual(channels);
    records.download(actual.data(), channels);
    if (expected_failure)
    {
      if (code != (((channels - 1) << 8) | 1))
        throw std::runtime_error("LU arithmetic failure channel changed");
      const auto& failed = actual.back();
      std::uint64_t bits;
      std::memcpy(&bits, &failed.log_abs, sizeof(bits));
      if (failed.status != 1 || cuCreal(failed.phase) || cuCimag(failed.phase) || bits != UINT64_C(0xfff0000000000000))
        throw std::runtime_error("Failed LU published regular metadata");
      ++comparisons;
      return;
    }
    if (code != clear)
      throw std::runtime_error("LU rejected finite oracle fixture");
    matrix.download(lu.data(), lu.size());
    for (std::size_t channel = 0; channel < channels; ++channel)
    {
      determinant::ComplexDeterminantWorkspace oracle(1, n);
      const auto expected = oracle.evaluate(input.data() + channel * n * n, nullptr, false);
      const auto& a = actual[channel];
      if (expected.amplitude.isZero())
      {
        std::uint64_t bits;
        std::memcpy(&bits, &a.log_abs, sizeof(bits));
        if (a.status != 1 || cuCreal(a.phase) || cuCimag(a.phase) || bits != UINT64_C(0xfff0000000000000))
          throw std::runtime_error("LU singular sentinel changed");
      }
      else
      {
        const double error = std::max(std::abs(a.log_abs - expected.amplitude.log_abs),
            std::abs(C(cuCreal(a.phase), cuCimag(a.phase)) - expected.amplitude.phase));
        max_error = std::max(max_error, error);
        if (a.status || !device_math::isFiniteBinary64(error) || error > 2e-10)
          throw std::runtime_error("LU independent CPU determinant mismatch");
      }
      ++comparisons;
    }
    // Same-device OFF/ON fingerprint detects changed pivot ties or arithmetic order.
    const auto* bytes = reinterpret_cast<const unsigned char*>(lu.data());
    for (std::size_t i = 0; i < lu.size() * sizeof(cuDoubleComplex); ++i)
      fingerprint = (fingerprint ^ bytes[i]) * 1099511628211ull;
    if (inverse)
    {
      inverse_buffer.download(inv.data(), inv.size());
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
        {
          C value{};
          for (std::size_t k = 0; k < n; ++k)
            value += input[i * n + k] * C(cuCreal(inv[k * n + j]), cuCimag(inv[k * n + j]));
          if (std::abs(value - C(i == j ? 1 : 0)) > 2e-12)
            throw std::runtime_error("LU inverse route changed");
        }
    }
  };

  if (growth_only)
  {
    // Wilkinson: unit diagonal/last column, -1 below diagonal, zero above.
    // First-row ties prevent swaps; the last column doubles at every pivot.
    // Finite (+/-1) inputs therefore overflow during the update at k=1023.
    // N1026 is deliberate: bounded N100 inputs cannot grow by 2^1024.
    constexpr std::size_t n = 1026;
    std::vector<C> growth(n * n);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        growth[i * n + j] = C(j == n - 1 || i == j ? 1 : (i > j ? -1 : 0));
    compare(n, growth, true);
    std::cout << "PASS finite_input_LU_elimination_growth Ne=1026 failure_channel=0 input_abs_max=1"
                 " expected_trailing_column_growth=2^1024 failed_metadata_unpublished=1\n";
    return;
  }

  // Compact dense tails, complex phases, exact pivot ties and singular nodes.
  for (const std::size_t n : {1, 3, 33, 100})
  {
    std::vector<C> dense(3 * n * n);
    for (std::size_t c = 0; c < 3; ++c)
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          dense[(c * n + i) * n + j] = C(i == j ? 2. + .01 * i : .005 * std::sin(double(i + 3 * j + c)),
              .003 * std::cos(double(2 * i + j + c)));
    compare(n, dense);
  }
  const std::vector<C> tie{{1, 0}, {2, 0}, {0, 0}, {0, 1}, {1, 0}, {1, 0}, {0, 0}, {0, 0}, {2, 0}};
  compare(3, tie);
  compare(3, tie, false, true);
  for (const double scale : {1e-200, 1e200})
  {
    auto scaled = tie;
    for (auto& z : scaled)
      z *= scale;
    compare(3, scaled);
  }
  compare(3, std::vector<C>(9));
  auto singular = tie;
  for (std::size_t j = 0; j < 3; ++j)
    singular[3 + j] = singular[j];
  compare(3, singular);
  auto invalid = tie;
  invalid.insert(invalid.end(), tie.begin(), tie.end());
  invalid.back() = {std::numeric_limits<double>::infinity(), 0};
  compare(3, invalid, true);
  const char* requested = std::getenv("PSIFORMER_GPU_COMPLEX_VALUE_COOPERATIVE_LU");
  std::cout << "PASS compact complex LU comparisons=" << comparisons << " max_CPU_error=" << max_error
            << " LU_fingerprint=" << fingerprint << " requested_cooperative=" << (requested ? requested : "0")
            << " inverse_route=serial\n";

  // Independent known determinants reproduce B's original component fixture.
  constexpr std::size_t n = 100, d = 16;
  std::vector<cuDoubleComplex> values(d * n * n);
  std::vector<C> phases(d, C{1});
  std::vector<double> logs(d);
  for (std::size_t c = 0; c < d; ++c)
    for (std::size_t i = 0; i < n; ++i)
    {
      const C diagonal(1. + .001 * (i + c), .003 * std::sin(double(i + 2 * c)));
      logs[c] += std::log(std::abs(diagonal));
      phases[c] *= diagonal / std::abs(diagonal);
      for (std::size_t j = i; j < n; ++j)
      {
        const C z = j == i ? diagonal : C(.001 * std::cos(double(i + j)), .0007);
        values[(c * n + (i + 1) % n) * n + j] = make_cuDoubleComplex(z.real(), z.imag());
      }
    }
  DeviceBuffer<cuDoubleComplex> matrix(values.size());
  DeviceBuffer<ComplexValueChannel> records(d);
  DeviceBuffer<std::uint64_t> status(1);
#ifdef QMC_CUDA2HIP
  using Event = hipEvent_t;
  const auto create = [](Event* e) { return hipEventCreate(e); };
  const auto record = [](Event e, Stream s) { return hipEventRecord(e, s); };
  const auto elapsed = [](float* t, Event a, Event b) { return hipEventElapsedTime(t, a, b); };
  const auto destroy = [](Event e) { return hipEventDestroy(e); };
#else
  using Event = cudaEvent_t;
  const auto create = [](Event* e) { return cudaEventCreate(e); };
  const auto record = [](Event e, Stream s) { return cudaEventRecord(e, s); };
  const auto elapsed = [](float* t, Event a, Event b) { return cudaEventElapsedTime(t, a, b); };
  const auto destroy = [](Event e) { return cudaEventDestroy(e); };
#endif
  Event begin, end;
  checkRuntime(create(&begin), "LU begin");
  checkRuntime(create(&end), "LU end");
  // Restore consumed input outside events. Keep all ten measurements after three warmups.
  for (int repetition = 0; repetition < 13; ++repetition)
  {
    const auto clear = std::numeric_limits<std::uint64_t>::max();
    matrix.upload(values.data(), values.size());
    status.upload(&clear, 1);
    checkRuntime(record(begin, queue.getNative()), "LU event");
    const auto start = std::chrono::steady_clock::now();
    checkRuntime(launchComplexValueFactor(queue.getNative(), d, n, matrix.data(), records.data(), status.data()), "timed LU");
    checkRuntime(record(end, queue.getNative()), "LU event");
    queue.sync();
    float ms = 0;
    checkRuntime(elapsed(&ms, begin, end), "LU elapsed");
    const auto wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::uint64_t code;
    status.download(&code, 1);
    std::vector<ComplexValueChannel> actual(d);
    records.download(actual.data(), d);
    if (code != clear)
      throw std::runtime_error("Timed LU status");
    for (std::size_t c = 0; c < d; ++c)
      if (actual[c].status || std::abs(actual[c].log_abs - logs[c]) > 2e-10 ||
          std::abs(C(cuCreal(actual[c].phase), cuCimag(actual[c].phase)) + phases[c]) > 2e-10)
        throw std::runtime_error("Timed LU known determinant mismatch");
    std::cout << "PASS component_VALUE_LU Ne=100 D=16 inverse=no repetition=" << repetition
              << " measured=" << (repetition >= 3) << " gpu_ms=" << ms << " wall_seconds=" << wall
              << " synthetic_triangle_not_physical_profile=1\n";
  }
  checkRuntime(destroy(begin), "LU destroy");
  checkRuntime(destroy(end), "LU destroy");
}
#endif
