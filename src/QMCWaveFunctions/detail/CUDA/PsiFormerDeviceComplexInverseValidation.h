//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_INVERSE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_COMPLEX_INVERSE_VALIDATION_H

/** @file PsiFormerDeviceComplexInverseValidation.h
 * Independent CPU and unchanged-device checks for private FULL inverse columns.
 * No physical Hamiltonian, sampling rule or optimizer tolerance is redefined.
 */
#include "PsiFormerDeviceOrbitalProjectionValidation.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerComplexDeterminant.h"
#include <cstring>

namespace qmcplusplus::psiformer::device
{
/// Preserve canonical factors, inverse failures and complete paired FULL products under column scheduling.
inline void validateComplexInverseColumns(bool cooperative_score_lu = false)
{
  using C = std::complex<double>;
  compute::Queue<PlatformKind::CUDA> queue;
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "inverse-column validation device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "inverse-column validation device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  std::size_t comparisons = 0;
  bool checked_cooperative_admission = false;
  double maximum_residual = 0, maximum_cpu = 0;
  const auto compare = [&](std::size_t n, const std::vector<C>& input, bool inverse_failure = false,
                           std::uint64_t incoming_status = std::numeric_limits<std::uint64_t>::max(),
                           bool nonunit_provider = false) {
    const auto channels = input.size() / (n * n);
    std::vector<cuDoubleComplex> packed(input.size()), sentinel(input.size(), make_cuDoubleComplex(37, -19));
    for (std::size_t i = 0; i < input.size(); ++i)
      packed[i] = make_cuDoubleComplex(input[i].real(), input[i].imag());
    DeviceBuffer<cuDoubleComplex> old_matrix(input.size()), new_matrix(input.size());
    DeviceBuffer<cuDoubleComplex> old_inverse(input.size()), new_inverse(input.size());
    DeviceBuffer<cuDoubleComplex> old_solve(channels * n), new_solve(channels * n);
    DeviceBuffer<std::size_t> old_permutation(channels * n), new_permutation(channels * n);
    DeviceBuffer<ComplexValueChannel> old_metadata(channels), new_metadata(channels);
    DeviceBuffer<std::uint64_t> old_status(1), new_status(1);
    DeviceBuffer<double> envelope(cooperative_score_lu ? input.size() : 0);
    DeviceBuffer<double> real(envelope.capacity()), imag(envelope.capacity());
    const auto clear = std::numeric_limits<std::uint64_t>::max();
    old_matrix.upload(packed.data(), packed.size());
    new_matrix.upload(packed.data(), packed.size());
    old_inverse.upload(sentinel.data(), sentinel.size());
    new_inverse.upload(sentinel.data(), sentinel.size());
    old_status.upload(&incoming_status, 1);
    new_status.upload(&incoming_status, 1);
    if (cooperative_score_lu)
    {
      const double provider_scale = nonunit_provider ? 2 : 1;
      std::vector<double> e(input.size(), provider_scale), r(input.size()), im(input.size());
      for (std::size_t i = 0; i < input.size(); ++i)
      {
        r[i] = input[i].real() / provider_scale;
        im[i] = input[i].imag() / provider_scale;
      }
      envelope.upload(e.data(), e.size());
      real.upload(r.data(), r.size());
      imag.upload(im.data(), im.size());
      if (!checked_cooperative_admission)
      {
        const auto before_admission = metrics.snapshot();
        for (int fault = 0; fault < 3; ++fault)
        {
          bool rejected = false;
          try
          {
            launchComplexScoreCooperativeInverseColumns(queue.getNative(), channels, fault == 2 ? 301 : n,
                fault == 0 ? nullptr : envelope.data(), real.data(), imag.data(), new_matrix.data(),
                new_metadata.data(), new_status.data(), fault == 1 ? new_matrix.data() : new_inverse.data(),
                new_permutation.data(), new_solve.data());
          }
          catch (const std::invalid_argument&) { rejected = true; }
          if (!rejected)
            throw std::runtime_error("Cooperative LU admitted null/alias/out-of-range metadata");
        }
        const auto after_admission = metrics.snapshot();
        if (before_admission.allocation_calls != after_admission.allocation_calls ||
            before_admission.uploaded_bytes != after_admission.uploaded_bytes)
          throw std::runtime_error("Cooperative LU invalid admission allocated or uploaded");
        checked_cooperative_admission = true;
      }
      const std::vector<cuDoubleComplex> solve_sentinel(channels * n, make_cuDoubleComplex(71, -13));
      old_solve.upload(solve_sentinel.data(), solve_sentinel.size());
      new_solve.upload(solve_sentinel.data(), solve_sentinel.size());
      // BOTH oracles start from the original pack, including NaN/signed-zero behavior.
      for (auto* matrix : {&old_matrix, &new_matrix})
        checkRuntime(launchComplexValuePack(queue.getNative(), input.size(), envelope.data(), real.data(),
            imag.data(), matrix->data(), matrix == &old_matrix ? old_status.data() : new_status.data()),
            "cooperative LU fixture original packed matrix");
      checkRuntime(launchComplexSpatialInverseColumns(queue.getNative(), channels, n, old_matrix.data(),
          old_metadata.data(), old_status.data(), old_inverse.data(), old_permutation.data(), old_solve.data()),
          "canonical scalar LU plus inverse columns");
    }
    else
      checkRuntime(launchComplexValueFactor(queue.getNative(), channels, n, old_matrix.data(), old_metadata.data(),
          old_status.data(), old_inverse.data(), old_permutation.data(), old_solve.data()), "old inverse oracle");
    queue.sync();
    const auto warm = metrics.snapshot();
    if (cooperative_score_lu)
      checkRuntime(launchComplexScoreCooperativeInverseColumns(queue.getNative(), channels, n, envelope.data(),
          real.data(), imag.data(), new_matrix.data(), new_metadata.data(), new_status.data(), new_inverse.data(),
          new_permutation.data(), new_solve.data()), "cooperative LU plus canonical failure replay");
    else
      checkRuntime(launchComplexSpatialInverseColumns(queue.getNative(), channels, n, new_matrix.data(), new_metadata.data(),
          new_status.data(), new_inverse.data(), new_permutation.data(), new_solve.data()), "parallel inverse oracle");
    queue.sync();
    const auto after = metrics.snapshot();
    if (after.allocation_calls != warm.allocation_calls || after.free_calls != warm.free_calls ||
        after.uploaded_bytes != warm.uploaded_bytes)
      throw std::runtime_error("Inverse columns allocated or uploaded HOST data during evaluation");
    std::uint64_t old_code, new_code;
    old_status.download(&old_code, 1);
    new_status.download(&new_code, 1);
    if (old_code != new_code)
      throw std::runtime_error("Inverse columns changed canonical failure channel");
    std::vector<cuDoubleComplex> a(input.size()), b(input.size());
    old_matrix.download(a.data(), a.size());
    new_matrix.download(b.data(), b.size());
    if (std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])))
      throw std::runtime_error("Inverse columns changed scalar LU bits");
    std::vector<ComplexValueChannel> old_records(channels), new_records(channels);
    old_metadata.download(old_records.data(), channels);
    new_metadata.download(new_records.data(), channels);
    if (std::memcmp(old_records.data(), new_records.data(), channels * sizeof(old_records[0])))
      throw std::runtime_error("Inverse columns changed phase/log/node metadata");
    std::vector<std::size_t> old_perm(channels * n), new_perm(channels * n);
    old_permutation.download(old_perm.data(), old_perm.size());
    new_permutation.download(new_perm.data(), new_perm.size());
    if (old_perm != new_perm)
      throw std::runtime_error("Inverse columns changed canonical pivot ties/permutations");
    if (cooperative_score_lu)
    {
      old_inverse.download(a.data(), a.size());
      new_inverse.download(b.data(), b.size());
      if (std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])))
        throw std::runtime_error("Cooperative LU inverse bits/failed prefix differ");
      std::vector<cuDoubleComplex> old_scale(channels * n), new_scale(channels * n);
      old_solve.download(old_scale.data(), old_scale.size());
      new_solve.download(new_scale.data(), new_scale.size());
      if (std::memcmp(old_scale.data(), new_scale.data(), old_scale.size() * sizeof(old_scale[0])))
        throw std::runtime_error("Cooperative LU scale/untouched solve-tail bits differ");
    }
    ++comparisons;
    // The legacy full-scalar versus columns oracle observes only status/metadata on inverse failure.
    if (old_code != clear)
    {
      if (inverse_failure && (old_code != 1 || new_records[0].status != 0))
        throw std::runtime_error("Finite inverse-overflow fixture did not retain regular LU metadata");
      return;
    }
    if (inverse_failure)
      throw std::runtime_error("Finite inverse-overflow fixture did not exercise inverse rejection");
    old_inverse.download(a.data(), a.size());
    new_inverse.download(b.data(), b.size());
    if (std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])))
      throw std::runtime_error("Inverse columns changed unchanged-GPU inverse bits");
    for (std::size_t channel = 0; channel < channels; ++channel)
    {
      determinant::ComplexDeterminantWorkspace cpu(1, n);
      const auto expected = cpu.evaluate(input.data() + channel * n * n, nullptr, false);
      if (expected.amplitude.isZero())
      {
        if (new_records[channel].status != 1)
          throw std::runtime_error("Inverse-column CPU node classification changed");
        continue;
      }
      const auto error = std::max(std::abs(new_records[channel].log_abs - expected.amplitude.log_abs),
          std::abs(C(cuCreal(new_records[channel].phase), cuCimag(new_records[channel].phase)) - expected.amplitude.phase));
      maximum_cpu = std::max(maximum_cpu, error);
      if (!std::isfinite(error) || error > 2e-10)
        throw std::runtime_error("Inverse-column CPU determinant exceeds existing LU contract");
      const auto offset = channel * n * n;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
        {
          C value{};
          for (std::size_t k = 0; k < n; ++k)
            value += input[offset + i * n + k] * C(cuCreal(b[offset + k * n + j]), cuCimag(b[offset + k * n + j]));
          const auto residual = std::abs(value - C(i == j ? 1 : 0));
          maximum_residual = std::max(maximum_residual, residual);
          if (!std::isfinite(residual) || residual > 2e-12)
            throw std::runtime_error("Inverse-column residual exceeds existing inverse contract");
          ++comparisons;
        }
    }
  };
  for (const std::size_t n : {std::size_t(2), std::size_t(64), std::size_t(100), std::size_t(200)})
  {
    std::vector<C> input(3 * n * n);
    for (std::size_t channel = 0; channel < 3; ++channel)
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          input[channel * n * n + i * n + j] = i == j ? C(2 + .001 * i, .05) :
              C(.01 * std::sin(.31 * (1 + i + 3 * j)), .007 * std::cos(.17 * (2 + 2 * i + j)));
    for (std::size_t j = 0; j < n; ++j)
      std::swap(input[n * n + j], input[n * n + (n - 1) * n + j]);
    for (std::size_t i = 2 * n * n; i < input.size(); ++i)
      input[i] *= n == 200 ? 1e120 : 1e-120;
    compare(n, input);
  }
  compare(2, {C(1), C(2), C(1), C(3)}); // Equal-magnitude pivot retains the first row.
  compare(2, std::vector<C>(4));
  auto invalid = std::vector<C>{C(2), C(0), C(0), C(3), C(1), C(0), C(0), C(1)};
  invalid[4] = C(std::numeric_limits<double>::quiet_NaN());
  compare(2, invalid);
  if (cooperative_score_lu)
    // Pivot squared norms=1e-220 stay normal; inverse(0,2)=1e330 overflows only in the solve.
    compare(3, {C(1e-110), C(1), C(0), C(0), C(1e-110), C(1), C(0), C(0), C(1e-110)}, true);
  else
    compare(2, {C(1e-310), C(0), C(0), C(1e-310)}, true);
  if (cooperative_score_lu)
  {
    compare(2, {C(2, -0.), C(0., -0.), C(-0., 0.), C(3, 0.)});
    compare(3, {C(1), C(2), C(3), C(0), C(1), C(2), C(0), C(0), C(0)});
    // Exact power-of-two providers exercise nonunit repack after a partial singular LU.
    compare(3, {C(1), C(2), C(3), C(0), C(1), C(2), C(0), C(0), C(0)}, false,
        std::numeric_limits<std::uint64_t>::max(), true);
    compare(2, {C(1), C(2), C(1), C(3)}, false, 7);
    compare(2, invalid, false, 7);
    auto infinite = invalid;
    infinite[4] = C(std::numeric_limits<double>::infinity());
    compare(2, infinite);
    // One cheap matrix-only maximum-size case covers row strides beyond the 256-thread block.
    std::vector<C> maximum(300 * 300);
    for (std::size_t i = 0; i < 300; ++i)
      for (std::size_t j = 0; j < 300; ++j)
        maximum[i * 300 + j] = i == j ? C(2 + .001 * i, .05) :
            C(.01 * std::sin(.31 * (1 + i + 3 * j)), .007 * std::cos(.17 * (2 + 2 * i + j)));
    compare(300, maximum);
  }
  const auto plan = makeTargetCostPlan(64, true, true);
  for (int fault = 0; fault < 6; ++fault)
  {
    auto invalid_environment = plan.environment();
    if (fault == 0) invalid_environment.periodic_profile = std::make_shared<PeriodicProfileMetadata>();
    if (fault == 1) invalid_environment.periodic_axes[1] = false;
    if (fault == 2) invalid_environment.fixed_nuclei = false;
    if (fault == 3) invalid_environment.head_representation = HeadRepresentation::REAL;
    if (fault == 4) invalid_environment.compute_scalar_domain = ScalarDomain::COMPLEX;
    if (fault == 5) invalid_environment.geometry_feature_policy = GeometryFeaturePolicy::PERIODIC_TWO_CELL_V1;
    if (supportsComplexInverseColumnsProfile(invalid_environment, 64))
      throw std::runtime_error("Inverse columns admitted unsupported metadata");
  }
  if (supportsComplexInverseColumnsProfile(plan.environment(), 63))
    throw std::runtime_error("Inverse columns admitted Ne below64");
  if (!cooperative_score_lu)
    validateBatchedHeadFullOwner(true);
  std::cout << std::setprecision(17) << (cooperative_score_lu ? "PASS cooperative score LU comparisons=" :
      "PASS inverse columns comparisons=") << comparisons
            << " max_CPU_determinant=" << maximum_cpu << " max_inverse_residual=" << maximum_residual
            << " old_GPU_LU_inverse_metadata_permutation_bits=exact warm_allocations=0 added_host_uploads=0\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
