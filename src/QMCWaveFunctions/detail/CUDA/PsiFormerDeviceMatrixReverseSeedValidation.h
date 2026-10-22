//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#ifndef QMCPLUSPLUS_PSIFORMER_DEVICE_MATRIX_REVERSE_SEED_VALIDATION_H
#define QMCPLUSPLUS_PSIFORMER_DEVICE_MATRIX_REVERSE_SEED_VALIDATION_H

#include "PsiFormerDeviceResources.h"
#include "PsiFormerDeviceDeterminantKernels.h"
#include "Platforms/CUDA/QueueCUDA.hpp"

#include <algorithm>
#include <cfenv>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qmcplusplus::psiformer::device
{
namespace matrix_reverse_seed_validation
{
constexpr const char* selector = "PSIFORMER_GPU_COOPERATIVE_MATRIX_REVERSE_SEEDS";

/// Isolate test-local selection and IEEE host fixtures from the caller's policy.
struct RestoreEnvironment
{
  bool existed = std::getenv(selector) != nullptr;
  std::string original = existed ? std::getenv(selector) : "";
  std::fenv_t floating;

  RestoreEnvironment()
  {
    if (std::fegetenv(&floating) != 0 || std::fesetenv(FE_DFL_ENV) != 0)
      throw std::runtime_error("Cannot establish the IEEE reverse-seed oracle");
  }

  ~RestoreEnvironment()
  {
    if (existed)
      ::setenv(selector, original.c_str(), 1);
    else
      ::unsetenv(selector);
    std::fesetenv(&floating);
  }
};

template<class T>
inline void requireBits(const std::vector<T>& actual, const std::vector<T>& expected, const char* what)
{
  if (actual.size() != expected.size() ||
      std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)) != 0)
    throw std::runtime_error(std::string("Matrix reverse-seed exact mismatch: ") + what);
}
} // namespace matrix_reverse_seed_validation

/// Compare every seed/status/guard against unchanged GPU and independent CPU maps.
inline void validateCooperativeMatrixReverseSeeds(compute::Queue<PlatformKind::CUDA>& queue)
{
  using namespace device_determinant;
  using namespace matrix_reverse_seed_validation;
  RestoreEnvironment restore;
  constexpr std::size_t guard = 3;
  constexpr double sentinel = 17.25;
  const auto status_sentinel = static_cast<DerivativeStatus>(0x5a);
  int ordinal = -1;
#ifdef QMC_CUDA2HIP
  checkRuntime(hipGetDevice(&ordinal), "reverse-seed validation device");
#else
  checkRuntime(cudaGetDevice(&ordinal), "reverse-seed validation device");
#endif
  auto& metrics = deviceMemoryMetrics(ordinal);
  std::size_t tested = 0;
  for (const std::size_t n : {2u, 63u, 64u, 65u, 100u, 200u, 300u})
  {
    const std::size_t batch = n == 64 ? 129 : (n == 100 ? 4 : 3);
    const std::size_t determinants = n == 64 ? 1 : (n == 100 ? 16 : 4);
    const auto matrix = n * n, span = batch * determinants * matrix;
    DeviceBuffer<FactorizationMetadata> d_factors;
    DeviceBuffer<CombinationMetadata> d_combination;
    DeviceBuffer<double> d_weights, d_inverse, d_old, d_new;
    DeviceBuffer<DerivativeStatus> d_old_status, d_new_status;
    d_factors.reserve(batch * determinants);
    d_combination.reserve(batch);
    d_weights.reserve(batch * determinants);
    d_inverse.reserve(span);
    d_old.reserve(span + 2 * guard);
    d_new.reserve(span + 2 * guard);
    d_old_status.reserve(batch + 2 * guard);
    d_new_status.reserve(batch + 2 * guard);
    for (std::size_t scenario = 0; scenario < 24; ++scenario)
    {
      std::vector<FactorizationMetadata> factors(batch * determinants);
      std::vector<CombinationMetadata> combinations(batch);
      std::vector<double> weights(batch * determinants), inverse(span);
      for (auto& factor : factors)
      {
        factor.status = FactorizationStatus::REGULAR;
        factor.inverse_available = true;
      }
      for (auto& combination : combinations)
      {
        combination.status = CombinationStatus::REGULAR;
        combination.phase = 1.0;
        combination.normalized_weights_available = true;
        combination.all_required_inverses_available = true;
      }
      for (std::size_t channel = 0; channel < weights.size(); ++channel)
        weights[channel] = channel % 2 ? -0.25 : 0.75;
      for (std::size_t element = 0; element < span; ++element)
        inverse[element] = (static_cast<int>((element * 17) % 29) - 14) / 32.0;
      const auto nan = std::numeric_limits<double>::quiet_NaN();
      switch (scenario)
      {
      case 1:
      case 2:
        weights[0] = scenario == 1 ? 0.0 : -0.0;
        std::fill(inverse.begin(), inverse.begin() + matrix, nan);
        factors[0].inverse_available = false;
        factors[0].status = FactorizationStatus::SINGULAR;
        break;
      case 3: combinations[0].status = CombinationStatus::NODE; weights[0] = nan; break;
      case 4: combinations[0].phase = 0.0; combinations[0].status = CombinationStatus::NONFINITE_INPUT; break;
      case 5: combinations[0].normalized_weights_available = false; break;
      case 6: combinations[0].all_required_inverses_available = false; weights[0] = nan; break;
      case 7: weights[0] = nan; factors[1].inverse_available = false; break;
      case 8: factors[0].inverse_available = false; weights[1] = nan; break;
      case 9: factors[0].status = FactorizationStatus::SINGULAR; break;
      case 10: factors[0].status = FactorizationStatus::NONFINITE_INPUT; weights[0] = 0.0; break;
      case 11: inverse[0] = nan; break;
      case 12: inverse[span - 1] = nan; break;
      case 13: inverse[matrix - 1] = std::numeric_limits<double>::infinity(); break;
      case 14: inverse[matrix] = -std::numeric_limits<double>::infinity(); break;
      case 15: weights[0] = 1e308; inverse[0] = 1e308; break;
      case 16: weights[0] = std::numeric_limits<double>::min(); inverse[0] = -std::numeric_limits<double>::min(); break;
      case 17: weights[0] = 2.0; inverse[0] = std::numeric_limits<double>::denorm_min(); break;
      case 18: inverse[0] = -0.0; inverse[1] = 0.0; break;
      case 19: weights[0] = std::numeric_limits<double>::infinity(); break;
      case 20: combinations[0].status = CombinationStatus::NONFINITE_INPUT; break;
      case 21:
        std::fill(weights.begin(), weights.end(), -0.0);
        std::fill(inverse.begin(), inverse.end(), nan);
        break;
      case 22: combinations[batch - 1].phase = 0.0; break;
      default: break; // Final regular case proves failure recovery on the same owners.
      }
      std::vector<double> expected(span + 2 * guard, sentinel), old(expected), actual(expected);
      std::vector<DerivativeStatus> expected_status(batch + 2 * guard, status_sentinel);
      auto old_status = expected_status, actual_status = expected_status;
      for (std::size_t configuration = 0; configuration < batch; ++configuration)
        expected_status[guard + configuration] = fillMatrixReverseSeeds(
            factors.data() + configuration * determinants, combinations[configuration],
            weights.data() + configuration * determinants, inverse.data() + configuration * determinants * matrix,
            determinants, n, expected.data() + guard + configuration * determinants * matrix);
      d_factors.upload(factors.data(), factors.size());
      d_combination.upload(combinations.data(), combinations.size());
      d_weights.upload(weights.data(), weights.size());
      d_inverse.upload(inverse.data(), inverse.size());
      d_old.upload(old.data(), old.size());
      d_new.upload(actual.data(), actual.size());
      d_old_status.upload(old_status.data(), old_status.size());
      d_new_status.upload(actual_status.data(), actual_status.size());
      const auto before = metrics.snapshot();
      for (const bool selected : {false, true})
      {
        ::setenv(selector, selected ? "1" : "0", 1);
        checkRuntime(launchDeterminantMatrixReverseSeeds(queue.getNative(), d_factors.data(),
            d_combination.data(), d_weights.data(), d_inverse.data(), batch, determinants, n,
            (selected ? d_new.data() : d_old.data()) + guard,
            (selected ? d_new_status.data() : d_old_status.data()) + guard), "reverse-seed map");
      }
      queue.sync();
      const auto after = metrics.snapshot();
      if (after.allocation_calls != before.allocation_calls || after.free_calls != before.free_calls ||
          after.live_bytes != before.live_bytes)
        throw std::runtime_error("Matrix reverse-seed map acquired hidden device owners");
      d_old.download(old.data(), old.size());
      d_new.download(actual.data(), actual.size());
      d_old_status.download(old_status.data(), old_status.size());
      d_new_status.download(actual_status.data(), actual_status.size());
      requireBits(old, expected, "original GPU/CPU including signed zero and guards");
      requireBits(actual, old, "candidate/original GPU including guards");
      requireBits(old_status, expected_status, "original status/CPU with byte guards");
      requireBits(actual_status, old_status, "candidate/original status with byte guards");
      if (n == 64 && scenario == 23)
      {
        for (const char* invalid : {"", "true", "01", " 1", "2"})
        {
          ::setenv(selector, invalid, 1);
          bool rejected = false;
          try
          {
            (void)launchDeterminantMatrixReverseSeeds(queue.getNative(), d_factors.data(),
                d_combination.data(), d_weights.data(), d_inverse.data(), batch, determinants, n,
                d_new.data() + guard, d_new_status.data() + guard);
          }
          catch (const std::invalid_argument&)
          {
            rejected = true;
          }
          if (!rejected)
            throw std::runtime_error("Matrix reverse-seed selector accepted malformed input");
        }
        ::setenv(selector, "1", 1);
        const auto warm = metrics.snapshot();
        for (int repeat = 0; repeat < 2; ++repeat)
          checkRuntime(launchDeterminantMatrixReverseSeeds(queue.getNative(), d_factors.data(),
              d_combination.data(), d_weights.data(), d_inverse.data(), batch, determinants, n,
              d_new.data() + guard, d_new_status.data() + guard), "reverse-seed warm recovery");
        queue.sync();
        if (metrics.snapshot().allocation_calls != warm.allocation_calls)
          throw std::runtime_error("Warmed reverse-seed map allocated");
        d_new.download(actual.data(), actual.size());
        requireBits(actual, expected, "warm recovery seeds");
        for (const std::size_t shift : {0u, 1u})
        {
          auto alias_expected = inverse;
          auto alias_actual = inverse;
          const auto expected_alias_status = fillMatrixReverseSeeds(factors.data(), combinations[0],
              weights.data(), alias_expected.data(), determinants, n, alias_expected.data() + shift);
          for (const bool selected : {false, true})
          {
            d_inverse.upload(inverse.data(), inverse.size());
            ::setenv(selector, selected ? "1" : "0", 1);
            checkRuntime(launchDeterminantMatrixReverseSeeds(queue.getNative(), d_factors.data(),
                d_combination.data(), d_weights.data(), d_inverse.data(), 1, determinants, n,
                d_inverse.data() + shift, d_new_status.data() + guard), "reverse-seed legacy alias fallback");
            queue.sync();
            d_inverse.download(alias_actual.data(), alias_actual.size());
            d_new_status.download(actual_status.data(), actual_status.size());
            requireBits(alias_actual, alias_expected, "legacy exact/partial in-place clear");
            if (actual_status[guard] != expected_alias_status)
              throw std::runtime_error("Matrix reverse-seed alias fallback changed status");
          }
        }
      }
      ++tested;
    }
  }
  ::setenv(selector, "invalid", 1);
  checkRuntime(launchDeterminantMatrixReverseSeeds(queue.getNative(), nullptr, nullptr, nullptr, nullptr,
      0, 0, 0, nullptr, nullptr), "reverse-seed zero population");
  std::cout << "PASS cooperative_matrix_reverse_seeds cases=" << tested
            << " exact_original_GPU_native_CPU signed_zero_nonfinite_overflow_status_precedence"
               " ignored_zero_weight_inverse guard_bytes zero_hidden_owner failed_request_recovery\n";
}
} // namespace qmcplusplus::psiformer::device
#endif
