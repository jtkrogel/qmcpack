//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_VALUE_SCALE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_VALUE_SCALE_VALIDATION_H

#include "PsiFormerDeviceResources.h"
#include "PsiFormerDeviceDeterminantKernels.h"
#include "Platforms/CUDA/QueueCUDA.hpp"

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qmcplusplus::psiformer::device
{
namespace value_scale_validation
{
/// Keep validation-local dispatch changes from escaping into other numerical cases.
struct RestoreOption
{
  std::string name;
  bool existed;
  std::string value;

  explicit RestoreOption(const char* key)
      : name(key), existed(std::getenv(key) != nullptr), value(existed ? std::getenv(key) : "")
  {
  }

  ~RestoreOption()
  {
    if (existed)
      ::setenv(name.c_str(), value.c_str(), 1);
    else
      ::unsetenv(name.c_str());
  }
};

/// Download one prepared owner after the caller has synchronized its queue.
template<class T>
std::vector<T> read(const DeviceBuffer<T>& owner)
{
  std::vector<T> result(owner.capacity());
  owner.download(result.data(), result.size());
  return result;
}

/// Require GPU bit parity or the established independent CPU floating-point bound.
inline void compareDouble(double actual, double expected, bool exact, const char* field)
{
  if (exact || !device_determinant::isFiniteBinary64(expected))
  {
    if (std::memcmp(&actual, &expected, sizeof(double)))
      throw std::runtime_error(std::string("VALUE scale exact mismatch: ") + field);
  }
  else if (!device_determinant::isFiniteBinary64(actual) ||
           std::abs(actual - expected) > 3e-11 * (1.0 + std::abs(expected)))
    throw std::runtime_error(std::string("VALUE scale native CPU mismatch: ") + field);
}

/// Check every public factorization field without comparing uninitialized struct padding.
inline void compareMetadata(const device_determinant::FactorizationMetadata& actual,
                            const device_determinant::FactorizationMetadata& expected, bool exact)
{
  if (actual.status != expected.status || actual.inverse_available != expected.inverse_available)
    throw std::runtime_error("VALUE scale changed factor/inverse admission");
  compareDouble(actual.matrix_scale, expected.matrix_scale, true, "scale");
  compareDouble(actual.phase, expected.phase, true, "phase");
  compareDouble(actual.log_abs, expected.log_abs, exact, "log_abs");
  compareDouble(actual.minimum_scaled_pivot, expected.minimum_scaled_pivot, exact, "minimum pivot");
  compareDouble(actual.maximum_scaled_pivot, expected.maximum_scaled_pivot, exact, "maximum pivot");
}
} // namespace value_scale_validation

/// Check the new scale map against its unchanged GPU map and native host factorization.
inline void validateCooperativeValueScale(compute::Queue<PlatformKind::CUDA>& queue)
{
  using namespace device_determinant;
  using namespace value_scale_validation;
  constexpr const char* selector = "PSIFORMER_GPU_VALUE_LU_PARALLEL_SCALE";
  RestoreOption restore_scale(selector);
  RestoreOption restore_inverse("PSIFORMER_GPU_COOPERATIVE_LU_INVERSE");
  // A fast-math executable startup may set host FTZ/DAZ even for this strict TU.
  // The independent CPU oracle must interpret the subnormal fixtures as IEEE FP64.
  /// Isolate the IEEE host oracle from process startup and restore the caller's state.
  struct RestoreFloatingEnvironment
  {
    std::fenv_t original;

    RestoreFloatingEnvironment()
    {
      if (std::fegetenv(&original) != 0 || std::fesetenv(FE_DFL_ENV) != 0)
        throw std::runtime_error("Cannot establish the native IEEE FP64 scale oracle");
    }

    ~RestoreFloatingEnvironment()
    {
      std::fesetenv(&original);
    }
  } restore_floating_environment;
  ::setenv("PSIFORMER_GPU_COOPERATIVE_LU_INVERSE", "1", 1);
  for (const std::size_t n : {2u, 63u, 64u, 65u, 100u, 200u, 300u})
  {
    constexpr std::size_t batch = 2, determinants = 10, count = batch * determinants;
    const auto elements = n * n;
    std::vector<double> input(count * elements);
    for (std::size_t m = 0; m < count; ++m)
    {
      double* a = input.data() + m * elements;
      for (std::size_t row = 0; row < n; ++row)
        for (std::size_t column = 0; column < n; ++column)
          a[row * n + column] = row == column ? 4.0 + (row % 3) / 8.0 :
              (static_cast<int>((row * 7 + column * 3) % 9) - 4) / (16.0 * n);
      if (m == 1)
        for (std::size_t column = 0; column < n; ++column)
          std::swap(a[column], a[(n - 1) * n + column]);
      if (m == 2 || m == 3 || m == 12 || m >= 14)
      {
        std::fill(a, a + elements, 0.0);
        for (std::size_t row = 0; row < n; ++row)
          a[row * n + row] = 1.0;
      }
      if (m == 2)
        a[(n - 1) * n] = -1.0; // Equal pivot magnitude must retain the earliest row.
      if (m == 3)
        a[0] = -1.0;
      if (m == 4)
        std::copy(a, a + n, a + (n - 1) * n);
      if (m == 5)
        for (std::size_t element = 0; element < elements; ++element)
          a[element] = element % 2 ? -0.0 : 0.0;
      if (m == 6)
        a[0] = std::numeric_limits<double>::quiet_NaN();
      if (m == 7)
        a[elements - 1] = std::numeric_limits<double>::quiet_NaN();
      if (m == 8 || m == 9)
        a[elements - 1] = m == 8 ? std::numeric_limits<double>::infinity() :
                                  -std::numeric_limits<double>::infinity();
      if (m == 9)
        a[0] = std::numeric_limits<double>::quiet_NaN();
      if (m == 10 || m == 11)
        for (std::size_t element = 0; element < elements; ++element)
          a[element] *= m == 10 ? 1e250 : 1e-250;
      if (m == 12)
      {
        a[1] = 4.0;
        a[elements - 1] = -4.0; // Maximum ties span independent scan lanes.
      }
      if (m == 13)
      {
        std::fill(a, a + elements, 0.0);
        const double base = std::ldexp(1.0, -830);
        for (std::size_t row = 0; row < n; ++row)
          a[row * n + row] = base;
        for (std::size_t row = 1; row < std::min(n, std::size_t(64)); ++row)
          a[row * n + row - 1] = -16 * base;
      }
      if (m == 14)
        a[elements - 1] = 0.0; // A late singular pivot preserves partial LU.
      if (m == 15 || m == 16 || m == 17)
      {
        const double scale = m == 15 ? std::numeric_limits<double>::max() :
            (m == 16 ? std::numeric_limits<double>::min() : std::numeric_limits<double>::denorm_min());
        for (std::size_t row = 0; row < n; ++row)
          a[row * n + row] = scale;
      }
      if (m == 18)
        a[elements - 1] = std::ldexp(1.0, -40);
      if (m == 19)
      {
        a[1] = -0.0;
        a[elements - 2] = -0.0;
      }
    }

    DeviceBuffer<double> matrices(input.size()), old_lu(input.size()), new_lu(input.size()),
        old_inverse(input.size()), new_inverse(input.size()), solve(count * n);
    DeviceBuffer<std::size_t> old_permutation(count * n), new_permutation(count * n);
    DeviceBuffer<FactorizationMetadata> old_metadata(count), new_metadata(count);
    matrices.upload(input.data(), input.size());
    std::vector<double> cpu_lu(elements), cpu_inverse(elements), cpu_solve(n);
    std::vector<std::size_t> cpu_permutation(n);

    for (const bool inverse : {false, true})
    {
      auto launch = [&](bool candidate) {
        ::setenv(selector, candidate ? "1" : "0", 1);
        auto& lu = candidate ? new_lu : old_lu;
        auto& inv = candidate ? new_inverse : old_inverse;
        auto& perm = candidate ? new_permutation : old_permutation;
        auto& metadata = candidate ? new_metadata : old_metadata;
        if (inverse)
          checkRuntime(launchDeterminantFactorization(queue.getNative(), matrices.data(), batch, determinants, n,
                           true, lu.data(), inv.data(), perm.data(), solve.data(), metadata.data()),
                       "scale inverse factorization");
        else
          checkRuntime(launchDeterminantValueFactorizationCooperative(queue.getNative(), matrices.data(), batch,
                           determinants, n, lu.data(), perm.data(), metadata.data()), "scale VALUE factorization");
      };
      const auto before = deviceMemoryMetrics(0).snapshot();
      launch(false);
      launch(true);
      queue.sync();
      const auto after = deviceMemoryMetrics(0).snapshot();
      if (before.allocation_calls != after.allocation_calls || before.live_bytes != after.live_bytes)
        throw std::runtime_error("VALUE scale scan allocated a hidden owner");
      const auto reference_lu = read(old_lu), actual_lu = read(new_lu);
      const auto reference_permutation = read(old_permutation), actual_permutation = read(new_permutation);
      const auto reference_metadata = read(old_metadata), actual_metadata = read(new_metadata);
      const auto reference_inverse = inverse ? read(old_inverse) : std::vector<double>{};
      const auto actual_inverse = inverse ? read(new_inverse) : std::vector<double>{};
      if (std::memcmp(reference_lu.data(), actual_lu.data(), actual_lu.size() * sizeof(double)) ||
          reference_permutation != actual_permutation ||
          (inverse && std::memcmp(reference_inverse.data(), actual_inverse.data(), actual_inverse.size() * sizeof(double))))
        throw std::runtime_error("VALUE scale scan changed exact GPU LU/permutation/inverse");
      for (std::size_t m = 0; m < count; ++m)
      {
        compareMetadata(actual_metadata[m], reference_metadata[m], true);
        const auto cpu = factorizeScaledReal(input.data() + m * elements, n, cpu_lu.data(), cpu_permutation.data(),
                                            inverse, cpu_inverse.data(), cpu_solve.data());
        compareMetadata(actual_metadata[m], cpu, false);
        for (std::size_t row = 0; row < n; ++row)
          if (actual_permutation[m * n + row] != cpu_permutation[row])
            throw std::runtime_error("VALUE scale scan changed native CPU permutation");
        for (std::size_t element = 0; element < elements; ++element)
        {
          compareDouble(actual_lu[m * elements + element], cpu_lu[element], false, "native LU");
          if (inverse)
            compareDouble(actual_inverse[m * elements + element], cpu_inverse[element], false, "native inverse");
        }
      }

      // A failed refresh must overwrite successful old contents, then recover exactly.
      auto failed = input;
      failed[0] = std::numeric_limits<double>::quiet_NaN();
      matrices.upload(failed.data(), failed.size());
      launch(true);
      queue.sync();
      const auto failed_lu = read(new_lu);
      const auto failed_metadata = read(new_metadata);
      if (failed_metadata[0].status != FactorizationStatus::NONFINITE_INPUT || failed_metadata[0].inverse_available ||
          std::any_of(failed_lu.begin(), failed_lu.begin() + elements, [](double x) { return x != 0.0; }))
        throw std::runtime_error("VALUE scale failed refresh retained old factorization");
      if (inverse)
      {
        const auto failed_inverse = read(new_inverse);
        if (std::any_of(failed_inverse.begin(), failed_inverse.begin() + elements, [](double x) { return x != 0.0; }))
          throw std::runtime_error("VALUE scale failed refresh retained old inverse");
      }
      matrices.upload(input.data(), input.size());
      launch(true);
      queue.sync();
      const auto recovered_lu = read(new_lu);
      const auto recovered_permutation = read(new_permutation);
      if (std::memcmp(reference_lu.data(), recovered_lu.data(), recovered_lu.size() * sizeof(double)) ||
          reference_permutation != recovered_permutation)
        throw std::runtime_error("VALUE scale recovery changed exact GPU factorization");
      const auto recovered_metadata = read(new_metadata);
      for (std::size_t m = 0; m < count; ++m)
        compareMetadata(recovered_metadata[m], reference_metadata[m], true);
      if (inverse)
      {
        const auto recovered_inverse = read(new_inverse);
        if (std::memcmp(reference_inverse.data(), recovered_inverse.data(), recovered_inverse.size() * sizeof(double)))
          throw std::runtime_error("VALUE scale recovery changed exact GPU inverse");
      }
      else
      {
        new_lu.upload(input.data(), input.size());
        checkRuntime(launchDeterminantValueFactorizationCooperative(queue.getNative(), new_lu.data(), batch,
                         determinants, n, new_lu.data(), new_permutation.data(), new_metadata.data()),
                     "scale in-place fallback");
        queue.sync();
        const auto alias_lu = read(new_lu);
        if (std::memcmp(reference_lu.data(), alias_lu.data(), alias_lu.size() * sizeof(double)) ||
            reference_permutation != read(new_permutation))
          throw std::runtime_error("VALUE scale changed exact in-place fallback");
        const auto alias_metadata = read(new_metadata);
        for (std::size_t m = 0; m < count; ++m)
          compareMetadata(alias_metadata[m], reference_metadata[m], true);
      }
      std::cout << "PASS cooperative_VALUE_scale Ne=" << n << " matrices=" << count << " inverse=" << inverse
                << " exact_GPU_LU_permutation_metadata_inverse native_CPU zero_hidden_owner failed_refresh_recovery\n";
    }
  }
}
} // namespace qmcplusplus::psiformer::device
#endif
