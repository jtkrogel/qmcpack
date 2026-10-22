//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_JET_REVERSE_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_JET_REVERSE_VALIDATION_H

#include "PsiFormerDeviceDeterminantKernels.h"
#include "PsiFormerDeviceResources.h"
#include "Platforms/QueueAliases.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace qmcplusplus::psiformer::device
{
/// Restore the caller's selector so numerical validation cannot change policy.
class JetReverseSelectorGuard
{
public:
  JetReverseSelectorGuard()
  {
    const char* value = std::getenv(name);
    present_ = value != nullptr;
    if (present_) value_ = value;
  }
  ~JetReverseSelectorGuard()
  {
    if (present_) ::setenv(name, value_.c_str(), 1);
    else ::unsetenv(name);
  }
  static constexpr const char* name = "PSIFORMER_GPU_COOPERATIVE_DETERMINANT_JET_REVERSE";
private:
  bool present_;
  std::string value_;
};

/// Keep the original complete CPU helper and original GPU kernel independent.
inline void validateCooperativeDeterminantJetReverse(compute::Queue<PlatformKind::CUDA>& queue)
{
  using device_determinant::DerivativeStatus;
  struct Shape { std::size_t n, k, d, b; };
  // Cover the admission boundary, padded target tails, actual D16/K1 and full jets.
  const Shape cases[]{{63,1,1,2},{64,1,2,2},{65,3,2,2},{100,1,16,1},
      {129,3,1,2},{200,1,2,1},{64,64,1,1}};
  JetReverseSelectorGuard restore;
  std::size_t checks = 0;
  for (const auto& shape : cases)
  {
    const auto elements = shape.d * shape.n * shape.n;
    const auto planes = 1 + 4 * shape.k;
    const auto begin = shape.k == shape.n ? 0 : shape.n - shape.k;
    const auto matrices = makeSpatialTargetJetLayout(shape.b, elements, shape.n, begin, shape.k,
        elements + 7, planes * (elements + 7) + 11);
    const auto roots = makeSpatialTargetJetLayout(shape.b, 1, shape.n, begin, shape.k,
        4, planes * 4 + 9);
    const LiftedDeterminantLayout layout{matrices, roots, shape.d, shape.n};
    const auto count = spatialJetSpanElements(matrices), root_count = spatialJetSpanElements(roots);
    DeviceBuffer<double> factors, root, old_output, new_output;
    DeviceBuffer<DerivativeStatus> factor_status, old_status, new_status;
    factors.reserve(count); root.reserve(root_count);
    old_output.reserve(count); new_output.reserve(count);
    factor_status.reserve(shape.b); old_status.reserve(shape.b); new_status.reserve(shape.b);
    for (unsigned int variant = 0; variant < (shape.n == 100 ? 16U : 15U); ++variant)
    {
      std::vector<double> host_factors(count, 19.5), host_root(root_count, -23.5), initial(count, 29.5);
      std::vector<DerivativeStatus> admission(shape.b, DerivativeStatus::AVAILABLE), expected_status(shape.b);
      for (std::size_t configuration = 0; configuration < shape.b; ++configuration)
        for (std::size_t plane = 0; plane < planes; ++plane)
        {
          host_root[roots.uncheckedPlaneOffset(configuration, plane, 0)] =
              (static_cast<int>((plane + 3 * configuration) % 9) - 4) / 32.0;
          for (std::size_t element = 0; element < elements; ++element)
            host_factors[matrices.uncheckedPlaneOffset(configuration, plane, element)] =
                (static_cast<int>((element * 7 + plane * 3 + configuration) % 23) - 11) / 64.0;
        }
      const auto last = shape.b - 1;
      const auto value_at = matrices.uncheckedValueOffset(last, elements - 1);
      const auto gradient_at = matrices.uncheckedGradientOffset(last, matrices.gradient_lanes - 1, elements - 1);
      const auto laplacian_at = matrices.uncheckedLaplacianOffset(last, shape.k - 1, elements - 1);
      const auto nan = std::numeric_limits<double>::quiet_NaN();
      // Root zeros do not revoke validation of any factor plane.
      if (variant == 1 || variant == 14)
        for (std::size_t plane = 0; plane < planes; ++plane)
          host_root[roots.uncheckedPlaneOffset(last, plane, 0)] = plane % 2 ? -0.0 : 0.0;
      if (variant == 2) host_factors[value_at] = nan;
      if (variant == 3 || variant == 14) host_factors[gradient_at] = nan;
      if (variant == 4) host_factors[laplacian_at] = nan;
      if (variant == 5) host_root[roots.uncheckedValueOffset(last, 0)] = nan;
      if (variant == 6) host_root[roots.uncheckedGradientOffset(last, 0, 0)] = nan;
      if (variant == 7) host_root[roots.uncheckedLaplacianOffset(last, 0, 0)] = nan;
      if (variant == 8)
      {
        host_root[roots.uncheckedValueOffset(last, 0)] = 2.0;
        host_factors[value_at] = 1e308;
      }
      if (variant == 9) admission[last] = DerivativeStatus::NODE;
      if (variant == 10) admission[last] = DerivativeStatus::REQUIRED_INVERSE_UNAVAILABLE;
      if (variant == 11) admission[last] = DerivativeStatus::NONFINITE_INPUT_OR_RESULT;
      if (variant == 12)
      {
        host_factors[value_at] = -0.0;
        host_factors[gradient_at] = 0x1p-1074;
        host_factors[laplacian_at] = -0x1p-1022;
      }
      if (variant == 15)
      {
        // Non-dyadic cancellation retains the tiny term through compensation.
        for (std::size_t plane = 0; plane < planes; ++plane)
          host_root[roots.uncheckedPlaneOffset(last, plane, 0)] = 1.0;
        for (std::size_t element = 0; element < elements; ++element)
        {
          host_factors[matrices.uncheckedValueOffset(last, element)] = 0.1;
          for (std::size_t lane = 0; lane < matrices.gradient_lanes; ++lane)
            host_factors[matrices.uncheckedGradientOffset(last, lane, element)] =
                lane % 3 == 0 ? 1e-20 : lane % 3 == 1 ? -0.1 : 0.3;
          for (std::size_t electron = 0; electron < shape.k; ++electron)
            host_factors[matrices.uncheckedLaplacianOffset(last, electron, element)] = -0.3;
        }
      }
      // Variant13 is a normal replay after all preceding failure fixtures.
      auto expected = initial;
      for (std::size_t configuration = 0; configuration < shape.b; ++configuration)
        expected_status[configuration] = device_determinant::reverseLiftedDeterminant(
            matrices, roots, configuration, admission[configuration], host_factors.data(),
            host_root.data(), expected.data());
      factors.upload(host_factors.data(), count); root.upload(host_root.data(), root_count);
      factor_status.upload(admission.data(), shape.b);
      old_output.upload(initial.data(), count); new_output.upload(initial.data(), count);
      ::setenv(JetReverseSelectorGuard::name, "0", 1);
      checkRuntime(launchDeterminantJetReverse(queue.getNative(), layout, factor_status.data(), factors.data(),
          root.data(), old_output.data(), old_status.data()), "unchanged jet reverse oracle");
      ::setenv(JetReverseSelectorGuard::name, "1", 1);
      checkRuntime(launchDeterminantJetReverse(queue.getNative(), layout, factor_status.data(), factors.data(),
          root.data(), new_output.data(), new_status.data()), "cooperative jet reverse candidate");
      queue.sync();
      std::vector<double> old_values(count), new_values(count);
      std::vector<DerivativeStatus> old_flags(shape.b), new_flags(shape.b);
      old_output.download(old_values.data(), count); new_output.download(new_values.data(), count);
      old_status.download(old_flags.data(), shape.b); new_status.download(new_flags.data(), shape.b);
      if (old_flags != expected_status || new_flags != expected_status ||
          std::memcmp(old_values.data(), new_values.data(), count * sizeof(double)) != 0 ||
          std::memcmp(old_values.data(), expected.data(), count * sizeof(double)) != 0)
        throw std::runtime_error("Cooperative determinant jet reverse changed CPU/GPU status, exact bits or padding: N=" +
            std::to_string(shape.n) + " K=" + std::to_string(shape.k) + " variant=" + std::to_string(variant));
      ++checks;
    }
    int ordinal = -1;
#ifdef QMC_CUDA2HIP
    checkRuntime(hipGetDevice(&ordinal), "jet reverse validation device");
#else
    checkRuntime(cudaGetDevice(&ordinal), "jet reverse validation device");
#endif
    const auto before = deviceMemoryMetrics(ordinal).snapshot();
    checkRuntime(launchDeterminantJetReverse(queue.getNative(), layout, factor_status.data(), factors.data(),
        root.data(), new_output.data(), new_status.data()), "warmed cooperative jet reverse");
    queue.sync();
    const auto after = deviceMemoryMetrics(ordinal).snapshot();
    if (before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls)
      throw std::runtime_error("Warmed determinant jet reverse allocated");
  }
  checkRuntime(launchDeterminantJetReverse(queue.getNative(), {}, nullptr, nullptr, nullptr, nullptr, nullptr),
      "empty determinant jet reverse");
  std::cout << "PASS cooperative determinant jet reverse: native_cases=" << checks
            << " independent CPU/GPU exact compensated sums statuses padding zero roots failures recovery warm_zero_alloc\n";
}
}
#endif
