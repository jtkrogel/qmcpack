//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_TARGET_FACTOR_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_TARGET_FACTOR_VALIDATION_H

/** @file PsiFormerDeviceComplexTargetFactorValidation.h
 * Check request-local factor decisions against the unchanged scalar device oracle.
 * Decision bytes are test-owned storage, never part of the production owner.
 */
#include "PsiFormerDeviceComplexInverseValidation.h"

namespace qmcplusplus::psiformer::device
{
/// Preserve exact factors, nodes and failure channels under guarded target reuse.
inline void validateComplexTargetFactorReuse()
{
  compute::Queue<PlatformKind::CUDA> queue;
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "target factor validation device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "target factor validation device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  std::size_t comparisons = 0, cold_count = 0, reused_count = 0, fallback_count = 0;
  double maximum_residual = 0;
  for (const std::size_t n : {std::size_t(2), std::size_t(64)})
  {
    constexpr std::size_t channels = 2;
    const auto elements = channels * n * n;
    std::vector<cuDoubleComplex> input(elements), sentinel(elements, make_cuDoubleComplex(37, -19));
    for (std::size_t c = 0; c < channels; ++c)
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          input[c * n * n + i * n + j] = make_cuDoubleComplex(i == j ? 2 + .01 * i :
              .003 * std::sin(.13 * (1 + i + 3 * j)), i == j ? 0 : .002 * std::cos(.17 * (2 + i + j)));
    // An explicit zero makes the signed-zero guard independent of libm values.
    input[1] = make_cuDoubleComplex(0, 0);
    DeviceBuffer<cuDoubleComplex> current(elements), canonical(elements), old_matrix(elements);
    DeviceBuffer<cuDoubleComplex> inverse(elements), old_inverse(elements), solve(channels * n), old_solve(channels * n);
    DeviceBuffer<std::size_t> permutation(channels * n), old_permutation(channels * n);
    DeviceBuffer<ComplexValueChannel> metadata(channels), old_metadata(channels);
    DeviceBuffer<std::uint64_t> status(1), old_status(1);
    DeviceBuffer<std::uint8_t> decisions(channels); // Exactly two diagnostic bytes, outside owner admission.
    inverse.upload(sentinel.data(), elements);
    const auto clear = std::numeric_limits<std::uint64_t>::max();

    const auto trial = [&](bool ready, const std::vector<std::uint8_t>& expected, bool expected_failure = false) {
      current.upload(input.data(), elements);
      old_matrix.upload(input.data(), elements);
      old_inverse.upload(sentinel.data(), elements);
      status.upload(&clear, 1);
      old_status.upload(&clear, 1);
      checkRuntime(launchComplexValueFactor(queue.getNative(), channels, n, old_matrix.data(), old_metadata.data(),
          old_status.data(), old_inverse.data(), old_permutation.data(), old_solve.data()), "canonical target factor oracle");
      queue.sync();
      const auto warm = metrics.snapshot();
      checkRuntime(launchComplexTargetFactorReuse(queue.getNative(), channels, n, current.data(), canonical.data(),
          ready, metadata.data(), status.data(), inverse.data(), permutation.data(), solve.data(), decisions.data()),
          "guarded target factor oracle");
      queue.sync();
      const auto after = metrics.snapshot();
      if (warm.allocation_calls != after.allocation_calls || warm.free_calls != after.free_calls ||
          warm.uploaded_bytes != after.uploaded_bytes)
        throw std::runtime_error("Target factors allocated or uploaded HOST data during their warm launch");
      std::vector<std::uint8_t> actual(channels);
      decisions.download(actual.data(), channels);
      if (actual != expected)
        throw std::runtime_error("Target factors did not take the exact cold/reuse/fallback decisions");
      for (auto decision : actual)
      {
        cold_count += decision == 0;
        reused_count += decision == 1;
        fallback_count += decision == 2;
      }
      std::uint64_t code = 0, old_code = 0;
      status.download(&code, 1);
      old_status.download(&old_code, 1);
      if (code != old_code || (code != clear) != expected_failure)
        throw std::runtime_error("Target factors changed canonical failure channel");
      std::vector<ComplexValueChannel> records(channels), old_records(channels);
      metadata.download(records.data(), channels);
      old_metadata.download(old_records.data(), channels);
      if (std::memcmp(records.data(), old_records.data(), channels * sizeof(records[0])))
        throw std::runtime_error("Target factors changed canonical phase/log/node metadata bits");
      std::vector<std::size_t> pivots(channels * n), old_pivots(channels * n);
      permutation.download(pivots.data(), pivots.size());
      old_permutation.download(old_pivots.data(), old_pivots.size());
      if (pivots != old_pivots)
        throw std::runtime_error("Target factors changed canonical pivot permutation");
      std::vector<cuDoubleComplex> raw(elements), scratch(elements), old_lu(elements);
      canonical.download(raw.data(), elements);
      current.download(scratch.data(), elements);
      old_matrix.download(old_lu.data(), elements);
      if (std::memcmp(raw.data(), input.data(), elements * sizeof(raw[0])))
        throw std::runtime_error("Target factors lost exact canonical raw matrix bits");
      for (std::size_t c = 0; c < channels; ++c)
        if (actual[c] != 1 && std::memcmp(scratch.data() + c * n * n, old_lu.data() + c * n * n,
            n * n * sizeof(scratch[0])))
          throw std::runtime_error("Target factors changed fallback scalar LU bits");
      ++comparisons;
      if (code != clear)
        return; // Failed inverse scratch is private and deliberately not compared.
      std::vector<cuDoubleComplex> result(elements), old_result(elements);
      inverse.download(result.data(), elements);
      old_inverse.download(old_result.data(), elements);
      for (std::size_t c = 0; c < channels; ++c)
        if (!records[c].status)
        {
          if (std::memcmp(result.data() + c * n * n, old_result.data() + c * n * n,
              n * n * sizeof(result[0])))
            throw std::runtime_error("Target factors changed unchanged-device inverse bits");
          for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
            {
              std::complex<double> value{};
              for (std::size_t k = 0; k < n; ++k)
              {
                const auto a = input[c * n * n + i * n + k], b = result[c * n * n + k * n + j];
                value += std::complex<double>(cuCreal(a), cuCimag(a)) *
                    std::complex<double>(cuCreal(b), cuCimag(b));
              }
              const auto error = std::abs(value - std::complex<double>(i == j ? 1 : 0));
              maximum_residual = std::max(maximum_residual, error);
              if (!std::isfinite(error) || error > 2e-12)
                throw std::runtime_error("Target factor residual exceeds existing inverse contract");
              ++comparisons;
            }
        }
    };
    trial(false, {0, 0});
    trial(true, {1, 1});
    input[0] = make_cuDoubleComplex(std::nextafter(cuCreal(input[0]), 3.), 0);
    trial(true, {2, 1});
    input[1] = make_cuDoubleComplex(-0., 0);
    trial(true, {2, 1}); // Floating equality is insufficient for the reuse guard.
    trial(false, {0, 0}); // A new coordinate request cannot inherit readiness.
    for (std::size_t i = 0; i < n * n; ++i) input[i] = make_cuDoubleComplex(0, 0);
    trial(true, {2, 1});
    trial(true, {2, 1}); // Singular metadata is never reused.
    input[0] = make_cuDoubleComplex(std::numeric_limits<double>::quiet_NaN(), 0);
    trial(true, {2, 1}, true);
    for (std::size_t i = 0; i < n * n; ++i) input[i] = make_cuDoubleComplex(0, 0);
    for (std::size_t i = 0; i < n; ++i) input[i * n + i] = make_cuDoubleComplex(1e-310, 0);
    trial(true, {2, 1}, true);
    trial(true, {2, 1}, true); // Regular LU metadata alone cannot bless a nonfinite inverse.
    bool alias_rejected = false;
    try { (void)launchComplexTargetFactorReuse(queue.getNative(), channels, n, current.data(), current.data(), false,
        metadata.data(), status.data(), inverse.data(), permutation.data(), solve.data()); }
    catch (const std::invalid_argument&) { alias_rejected = true; }
    if (!alias_rejected)
      throw std::runtime_error("Target factor alias admission unexpectedly succeeded");
  }
  std::cout << std::setprecision(17) << "PASS target factors comparisons=" << comparisons
            << " cold_channels=" << cold_count << " reused_channels=" << reused_count
            << " fallback_channels=" << fallback_count << " maximum_inverse_residual=" << maximum_residual
            << " old_GPU_LU_inverse_metadata_permutation_bits=exact signed_zero_guard=exact"
            << " test_only_decision_bytes=2 production_added_bytes=0 warm_allocations=0 added_HOST_uploads=0\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
