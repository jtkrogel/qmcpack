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
  struct ShapeCase { bool cooperative_cold; std::size_t n, channels; };
  constexpr ShapeCase cases[]{{false, 2, 2}, {false, 64, 2}, {false, 200, 32},
                             {true, 2, 2}, {true, 64, 2}, {true, 200, 32}};
  for (const auto shape : cases)
  {
    const auto cooperative_cold = shape.cooperative_cold;
    const auto n = shape.n, channels = shape.channels;
    const auto elements = channels * n * n;
    std::vector<cuDoubleComplex> input(elements), sentinel(elements, make_cuDoubleComplex(37, -19));
    for (std::size_t c = 0; c < channels; ++c)
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          input[c * n * n + i * n + j] = make_cuDoubleComplex(i == j ? 2 + .01 * i :
              .003 * std::sin(.13 * (1 + i + 3 * j)), i == j ? 0 : .002 * std::cos(.17 * (2 + i + j)));
    // An explicit zero makes the signed-zero guard independent of libm values.
    input[1] = make_cuDoubleComplex(0, 0);
    if (n == 200)
    {
      // A nonzero tie straddles both vendor warp widths; the lowest row must win.
      input[n * n + 33 * n] = input[n * n + 65 * n] = make_cuDoubleComplex(4, 0);
    }
    const auto finite_input = input;
    DeviceBuffer<cuDoubleComplex> current(elements), canonical(elements), old_matrix(elements), old_raw(elements);
    DeviceBuffer<cuDoubleComplex> inverse(elements), old_inverse(elements), solve(channels * n), old_solve(channels * n);
    DeviceBuffer<cuDoubleComplex> column_matrix(elements), column_inverse(elements), column_solve(channels * n);
    DeviceBuffer<std::size_t> permutation(channels * n), old_permutation(channels * n);
    DeviceBuffer<std::size_t> column_permutation(channels * n);
    DeviceBuffer<ComplexValueChannel> metadata(channels), old_metadata(channels);
    DeviceBuffer<ComplexValueChannel> column_metadata(channels);
    DeviceBuffer<std::uint64_t> status(1), old_status(1);
    DeviceBuffer<std::uint64_t> column_status(1);
    DeviceBuffer<std::uint8_t> decisions(channels); // Two or32 test-only bytes, outside owner admission.
    inverse.upload(sentinel.data(), elements);
    const auto clear = std::numeric_limits<std::uint64_t>::max();

    const auto trial = [&](bool ready, const std::vector<std::uint8_t>& expected, int expected_failure = 0) {
      current.upload(input.data(), elements);
      old_matrix.upload(input.data(), elements);
      old_inverse.upload(sentinel.data(), elements);
      status.upload(&clear, 1);
      old_status.upload(&clear, 1);
      checkRuntime(launchComplexTargetFactorReuse(queue.getNative(), channels, n, old_matrix.data(), old_raw.data(),
          false, old_metadata.data(), old_status.data(), old_inverse.data(), old_permutation.data(), old_solve.data()),
          "unchanged cold target factor oracle");
      if (cooperative_cold && !ready)
      {
        const std::vector<cuDoubleComplex> scratch(channels * n, make_cuDoubleComplex(37, -19));
        column_matrix.upload(input.data(), elements);
        column_inverse.upload(sentinel.data(), elements);
        inverse.upload(sentinel.data(), elements);
        column_solve.upload(scratch.data(), scratch.size());
        solve.upload(scratch.data(), scratch.size());
        column_status.upload(&clear, 1);
        checkRuntime(launchComplexSpatialInverseColumns(queue.getNative(), channels, n, column_matrix.data(),
            column_metadata.data(), column_status.data(), column_inverse.data(), column_permutation.data(),
            column_solve.data()), "qualified scalar LU/column scratch oracle");
      }
      queue.sync();
      const auto warm = metrics.snapshot();
      if (cooperative_cold && !ready)
        checkRuntime(launchComplexTargetCooperativeColdFactor(queue.getNative(), channels, n, current.data(),
            canonical.data(), metadata.data(), status.data(), inverse.data(), permutation.data(), solve.data(),
            decisions.data()), "private cooperative cold target factor");
      else
        checkRuntime(launchComplexTargetFactorReuse(queue.getNative(), channels, n, current.data(), canonical.data(),
            ready, metadata.data(), status.data(), inverse.data(), permutation.data(), solve.data(), decisions.data()),
            "unchanged guarded target factor");
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
      if (code != old_code || (expected_failure >= 0 && (code != clear) != bool(expected_failure)))
        throw std::runtime_error("Target factors changed canonical failure channel");
      if (expected_failure < 0)
        std::cout << "target_finite_subnormal_probe n=" << n << " cooperative_cold=" << cooperative_cold
                  << " unchanged_oracle_status=" << old_code << " candidate_status=" << code << '\n';
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
      if (n == 200 && pivots[n] != 33)
        throw std::runtime_error("Cold target factor changed the cross-warp lowest-row tie");
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
      if (cooperative_cold && !ready)
      {
        std::uint64_t column_code;
        column_status.download(&column_code, 1);
        if (column_code != code)
          throw std::runtime_error("Cold target factor changed the qualified inverse failure status");
        std::vector<cuDoubleComplex> a(elements), b(elements), sa(channels * n), sb(channels * n);
        inverse.download(a.data(), elements); column_inverse.download(b.data(), elements);
        solve.download(sa.data(), sa.size()); column_solve.download(sb.data(), sb.size());
        if (std::memcmp(a.data(), b.data(), elements * sizeof(a[0])) ||
            std::memcmp(sa.data(), sb.data(), sa.size() * sizeof(sa[0])))
          throw std::runtime_error("Cold target factor changed column failed-prefix/scale/solve-tail bits");
      }
      ++comparisons;
      if (code != clear)
        return; // Legacy failed inverse scratch is private; the cold column-prefix oracle was checked above.
      std::vector<cuDoubleComplex> result(elements), old_result(elements);
      inverse.download(result.data(), elements);
      old_inverse.download(old_result.data(), elements);
      for (std::size_t c = 0; c < channels; ++c)
        if (!records[c].status)
        {
          if (std::memcmp(result.data() + c * n * n, old_result.data() + c * n * n,
              n * n * sizeof(result[0])))
            throw std::runtime_error("Target factors changed unchanged-device inverse bits");
          for (std::size_t i = 0; n != 200 && i < n; ++i)
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
    trial(false, std::vector<std::uint8_t>(channels, 0));
    trial(true, std::vector<std::uint8_t>(channels, 1));
    if (n == 200)
      continue; // Full GPU bit oracles above, without a new large cubic CPU residual test.
    input[0] = make_cuDoubleComplex(std::nextafter(cuCreal(input[0]), 3.), 0);
    trial(true, {2, 1});
    input[1] = make_cuDoubleComplex(-0., 0);
    trial(true, {2, 1}); // Floating equality is insufficient for the reuse guard.
    trial(false, {0, 0}); // A new coordinate request cannot inherit readiness.
    for (const int exponent : {-500, 500})
    {
      for (std::size_t i = 0; i < elements; ++i)
        input[i] = make_cuDoubleComplex(std::ldexp(cuCreal(finite_input[i]), exponent),
            std::ldexp(cuCimag(finite_input[i]), exponent));
      trial(false, {0, 0});
      trial(true, {1, 1});
    }
    for (std::size_t i = 0; i < n * n; ++i) input[i] = make_cuDoubleComplex(0, 0);
    trial(false, {0, 0});
    trial(true, {2, 1});
    trial(true, {2, 1}); // Singular metadata is never reused.
    const std::uint64_t nan_payload = 0x7ff8000000000042ull;
    double payload_value;
    std::memcpy(&payload_value, &nan_payload, sizeof(payload_value));
    input[0] = make_cuDoubleComplex(payload_value, -0.0);
    trial(false, {0, 0}, true);
    trial(true, {2, 1}, true);
    input[0] = make_cuDoubleComplex(std::numeric_limits<double>::infinity(), -0.0);
    trial(false, {0, 0}, true);
    trial(true, {2, 1}, true);
    input = finite_input;
    for (std::size_t i = 0; i < n; ++i)
      input[i * n] = make_cuDoubleComplex(std::ldexp(1.0, -1070), 0);
    trial(false, {0, 0}, -1); // Vendor-specific finite-subnormal outcome, exact within-vendor oracle.
    for (std::size_t i = 0; i < n * n; ++i) input[i] = make_cuDoubleComplex(0, 0);
    for (std::size_t i = 0; i < n; ++i) input[i * n + i] = make_cuDoubleComplex(1e-310, 0);
    trial(false, {0, 0}, true);
    trial(true, {2, 1}, true);
    trial(true, {2, 1}, true); // Regular LU metadata alone cannot bless a nonfinite inverse.
    bool alias_rejected = false;
    try { (void)launchComplexTargetFactorReuse(queue.getNative(), channels, n, current.data(), current.data(), false,
        metadata.data(), status.data(), inverse.data(), permutation.data(), solve.data()); }
    catch (const std::invalid_argument&) { alias_rejected = true; }
    if (!alias_rejected)
      throw std::runtime_error("Target factor alias admission unexpectedly succeeded");
    if (cooperative_cold)
    {
      const auto before = metrics.snapshot();
      const auto reject = [&](auto invoke) {
        bool rejected = false;
        try { invoke(); }
        catch (const std::invalid_argument&) { rejected = true; }
        catch (const std::overflow_error&) { rejected = true; }
        catch (const std::length_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("Cold target factor accepted invalid prelaunch metadata");
      };
      const auto call = [&](std::size_t count, std::size_t size, cuDoubleComplex* raw,
                            cuDoubleComplex* work, std::uint8_t* decision) {
        return launchComplexTargetCooperativeColdFactor(queue.getNative(), count, size, current.data(), raw,
            metadata.data(), status.data(), inverse.data(), permutation.data(), work, decision);
      };
      reject([&] { call(channels, n, current.data() + 1, solve.data(), decisions.data()); });
      reject([&] { call(channels, n, canonical.data(), nullptr, decisions.data()); });
      reject([&] { call(channels, 0, canonical.data(), solve.data(), decisions.data()); });
      reject([&] { call(channels, 301, canonical.data(), solve.data(), decisions.data()); });
      reject([&] { call(std::numeric_limits<std::size_t>::max(), n, canonical.data(), solve.data(), decisions.data()); });
      reject([&] { call(channels, n, canonical.data(), solve.data(), reinterpret_cast<std::uint8_t*>(status.data())); });
      const auto after = metrics.snapshot();
      if (before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls ||
          before.uploaded_bytes != after.uploaded_bytes || before.downloaded_bytes != after.downloaded_bytes ||
          before.device_copy_calls != after.device_copy_calls)
        throw std::runtime_error("Rejected cold target metadata changed resource counters");
    }
  }
  std::cout << std::setprecision(17) << "PASS target factors comparisons=" << comparisons
            << " cold_channels=" << cold_count << " reused_channels=" << reused_count
            << " fallback_channels=" << fallback_count << " maximum_inverse_residual=" << maximum_residual
            << " old_GPU_LU_inverse_metadata_permutation_bits=exact signed_zero_guard=exact"
            << " cooperative_cold_then_unchanged_reuse=exact column_failed_prefix_scale=exact"
            << " test_only_decision_bytes_max=32 production_added_bytes=0 warm_allocations=0 added_HOST_uploads=0\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
