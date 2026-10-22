//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceRuntimeValidation.cu
 * @brief Opt-in actual CUDA/HIP numerical validation, separate from offline tests.
 *
 * This executable launches nonzero device work and compares downloaded outputs
 * with independent CPU formulas. It cannot certify a complete PsiFormer executor.
 * No device is exit 77, never a passing GPU test. --host-check avoids runtime calls.
 */

#include "PsiFormerDeviceResources.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerDeviceMath.h"
#include "PsiFormerDeviceKernels.h"
#include "PsiFormerDeviceDense.h"
#include "PsiFormerDeviceSpatial.h"
#include "PsiFormerDeviceDerivativeKernels.h"
#include "PsiFormerDeviceRealVectorValidation.h"
#include "PsiFormerDeviceAttentionJetValidation.h"
#include "PsiFormerDeviceValueCuspValidation.h"
#include "PsiFormerDeviceDeterminantKernels.h"
#include "PsiFormerDeviceValueScaleValidation.h"
#include "Platforms/CUDA/AccelBLAS_CUDA.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace
{
using namespace qmcplusplus::psiformer;
using namespace qmcplusplus::psiformer::device;
using Queue = qmcplusplus::compute::Queue<qmcplusplus::PlatformKind::CUDA>;

/// Compare finite scalars with combined absolute/relative FP64 tolerances.
void near(double actual, double expected, const char* field, double tolerance = 3e-11)
{
  if (!device_math::isFiniteBinary64(actual) || !device_math::isFiniteBinary64(expected) ||
      std::abs(actual - expected) > tolerance * (1.0 + std::abs(expected)))
    throw std::runtime_error(std::string(field) + ": actual=" + std::to_string(actual) +
                             " expected=" + std::to_string(expected));
}

/// Generate reproducible modest values without vendor-specific random distributions.
std::vector<double> randomValues(std::size_t size, std::uint32_t seed)
{
  std::mt19937 generator(seed);
  std::vector<double> values(size);
  for (double& value : values)
    value = (static_cast<int>(generator() % 1025) - 512) / 1024.0;
  return values;
}

/// Upload an owning host vector into a new explicitly prepared device buffer.
template<class T>
DeviceBuffer<T> upload(const std::vector<T>& values)
{
  DeviceBuffer<T> buffer(values.size());
  buffer.upload(values.data(), values.size());
  return buffer;
}

/// Download a complete prepared buffer after its stream has been synchronized.
template<class T>
std::vector<T> download(const DeviceBuffer<T>& buffer)
{
  std::vector<T> values(buffer.capacity());
  buffer.download(values.data(), values.size());
  return values;
}

/// Exercise overflow and zero-capacity invariants without requiring a device.
void hostChecks()
{
  DeviceBuffer<double> empty;
  empty.upload(nullptr, 0);
  empty.download(nullptr, 0);
  empty.reserve(0);
  if (empty.data() || empty.capacity())
    throw std::runtime_error("zero-capacity buffer invariant");
  bool overflow = false;
  try { (void)checkedDeviceBytes<double>(std::numeric_limits<std::size_t>::max()); }
  catch (const std::overflow_error&) { overflow = true; }
  if (!overflow)
    throw std::runtime_error("byte overflow was not rejected");
  bool bounds = false;
  try { empty.upload(nullptr, 1); }
  catch (const std::length_error&) { bounds = true; }
  if (!bounds)
    throw std::runtime_error("capacity bound was not rejected");
  bool rejected_nonfinite = false;
  try { near(std::numeric_limits<double>::quiet_NaN(), 0, "nonfinite host guard"); }
  catch (const std::runtime_error&) { rejected_nonfinite = true; }
  if (!rejected_nonfinite)
    throw std::runtime_error("nonfinite comparison guard was optimized away");
}

/// Check VALUE-only launch admission without dereferencing host dummy addresses.
void valueLuHostChecks()
{
  using device_determinant::FactorizationMetadata;
  double value = 1;
  std::size_t permutation = 0;
  FactorizationMetadata metadata;
  auto invoke = [&](std::size_t b, std::size_t d, std::size_t n, const double* input,
                    double* lu, std::size_t* perm, FactorizationMetadata* meta) {
    return launchDeterminantValueFactorizationCooperative(nullptr, input, b, d, n, lu, perm, meta);
  };
  checkRuntime(invoke(0, 1, 0, nullptr, nullptr, nullptr, nullptr), "zero VALUE batch");
  checkRuntime(invoke(1, 0, 0, nullptr, nullptr, nullptr, nullptr), "zero VALUE determinants");
  auto invalid = [&](auto call) {
    bool rejected = false;
    try { call(); } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("VALUE launch failed to reject invalid arguments");
  };
  invalid([&] { invoke(1, 1, 0, &value, &value, &permutation, &metadata); });
  invalid([&] { invoke(1, 1, 2, nullptr, &value, &permutation, &metadata); });
  invalid([&] { invoke(1, 1, 2, &value, nullptr, &permutation, &metadata); });
  invalid([&] { invoke(1, 1, 2, &value, &value, nullptr, &metadata); });
  invalid([&] { invoke(1, 1, 2, &value, &value, &permutation, nullptr); });
  auto overflow = [&](std::size_t b, std::size_t d, std::size_t n) {
    bool rejected = false;
    double separate_lu = 0;
    try { invoke(b, d, n, &value, &separate_lu, &permutation, &metadata); }
    catch (const std::length_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("VALUE launch failed to reject layout overflow");
  };
  const auto maximum = std::numeric_limits<std::size_t>::max();
  overflow(maximum, 2, 64);
  overflow(1, 1, maximum);
  overflow(maximum / 4096 + 1, 1, 64);
  overflow(static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1, 1, 64);
}

/// Compare every LU entry, permutation and metadata field, including singular partial LU.
void valueLuCases(Queue& queue)
{
  using namespace device_determinant;
  auto same = [](double actual, double expected, const char* field, bool exact) {
    if (exact)
    {
      if (std::memcmp(&actual, &expected, sizeof(double)) != 0)
        throw std::runtime_error(std::string(field) + " bitwise fallback mismatch");
    }
    else if (!device_math::isFiniteBinary64(expected))
    {
      if (std::memcmp(&actual, &expected, sizeof(double)) != 0)
        throw std::runtime_error(std::string(field) + " nonfinite metadata mismatch");
    }
    else
      near(actual, expected, field);
  };
  for (const std::size_t n : {2u, 63u, 64u, 65u, 100u, 200u, 300u})
  {
    // Ne300 cheaply exercises row/multiplier loops striding past all 128 threads.
    const std::size_t b = n == 300 ? 1 : 2, d = n == 300 ? 1 : 6, count = b * d;
    std::vector<double> input(count * n * n);
    for (std::size_t m = 0; m < count; ++m)
    {
      double* a = input.data() + m * n * n;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          a[i * n + j] = i == j ? 4.0 + (i % 3) / 8.0 :
              (static_cast<int>((i * 7 + j * 3) % 9) - 4) / (16.0 * n);
      if (m == 1)
        for (std::size_t j = 0; j < n; ++j) std::swap(a[j], a[(n - 1) * n + j]);
      if (m == 2 || m == 3 || m == 11)
      {
        std::fill(a, a + n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) a[i * n + i] = 1;
        if (m == 2) { a[n] = 1; if (n > 2) a[2 * n] = -1; }
        if (m == 3) a[0] = -1;
        if (m == 11) a[(n - 1) * n + n - 1] = std::ldexp(1.0, -40);
      }
      if (m == 4) std::copy(a, a + n, a + (n - 1) * n);
      if (m == 5) std::fill(a, a + n * n, 0.0);
      if (m == 6) a[n + 1] = std::numeric_limits<double>::quiet_NaN();
      if (m == 7) a[n + 1] = std::numeric_limits<double>::infinity();
      if (m == 8) a[n + 1] = -std::numeric_limits<double>::infinity();
      if (m == 9 || m == 10)
        for (std::size_t i = 0; i < n * n; ++i) a[i] *= m == 9 ? 1e250 : 1e-250;
    }
    std::vector<double> cpu_lu(input.size());
    std::vector<std::size_t> cpu_perm(count * n);
    std::vector<FactorizationMetadata> cpu_meta(count);
    for (std::size_t m = 0; m < count; ++m)
      cpu_meta[m] = factorizeScaledReal(input.data() + m * n * n, n,
          cpu_lu.data() + m * n * n, cpu_perm.data() + m * n, false, nullptr, nullptr);
    auto matrices = upload(input);
    DeviceBuffer<double> scalar_lu(input.size()), cooperative_lu(input.size());
    DeviceBuffer<std::size_t> scalar_perm(count * n), cooperative_perm(count * n);
    DeviceBuffer<FactorizationMetadata> scalar_meta(count), cooperative_meta(count);
    checkRuntime(launchDeterminantFactorization(queue.getNative(), matrices.data(), b, d, n, false,
        scalar_lu.data(), nullptr, scalar_perm.data(), nullptr, scalar_meta.data()), "scalar VALUE reference");
    checkRuntime(launchDeterminantValueFactorizationCooperative(queue.getNative(), matrices.data(), b, d, n,
        cooperative_lu.data(), cooperative_perm.data(), cooperative_meta.data()), "cooperative VALUE");
    queue.sync();
    const auto old_lu = download(scalar_lu);
    const auto old_perm = download(scalar_perm);
    const auto old_meta = download(scalar_meta);
    auto compare = [&](const auto& lu, const auto& perm, const auto& meta,
                       const auto& reference_lu, const auto& reference_perm, const auto& reference_meta,
                       bool exact) {
      for (std::size_t i = 0; i < lu.size(); ++i) same(lu[i], reference_lu[i], "VALUE LU", exact);
      if (perm != reference_perm) throw std::runtime_error("VALUE permutation mismatch");
      for (std::size_t m = 0; m < count; ++m)
      {
        const auto& actual = meta[m]; const auto& expected = reference_meta[m];
        if (actual.status != expected.status || actual.inverse_available != expected.inverse_available ||
            actual.phase != expected.phase)
          throw std::runtime_error("VALUE status/phase/inverse mismatch");
        same(actual.log_abs, expected.log_abs, "VALUE logabs", exact);
        same(actual.matrix_scale, expected.matrix_scale, "VALUE scale", exact);
        same(actual.minimum_scaled_pivot, expected.minimum_scaled_pivot, "VALUE min pivot", exact);
        same(actual.maximum_scaled_pivot, expected.maximum_scaled_pivot, "VALUE max pivot", exact);
      }
    };
    const auto new_lu = download(cooperative_lu);
    const auto new_perm = download(cooperative_perm);
    const auto new_meta = download(cooperative_meta);
    compare(new_lu, new_perm, new_meta, old_lu, old_perm, old_meta, n < 64);
    compare(new_lu, new_perm, new_meta, cpu_lu, cpu_perm, cpu_meta, false);
    if (count > 1 && (new_perm[n] != n - 1 || new_perm[2 * n] != 0 || new_meta[3].phase != -1 ||
        new_meta[4].status != FactorizationStatus::SINGULAR || new_lu[4 * n * n] == 0))
      throw std::runtime_error("VALUE edge fixture lost swap/tie/sign/partial singular premise");
    // Exact input/output alias must select the unchanged scalar implementation.
    auto alias = upload(input);
    checkRuntime(launchDeterminantValueFactorizationCooperative(queue.getNative(), alias.data(), b, d, n,
        alias.data(), cooperative_perm.data(), cooperative_meta.data()), "aliased scalar VALUE fallback");
    queue.sync();
    compare(download(alias), download(cooperative_perm), download(cooperative_meta),
            old_lu, old_perm, old_meta, true);
    std::cout << "PASS cooperative_VALUE_LU Ne=" << n << " matrices=" << count
              << " all_LU_permutation_metadata CPU_scalar_reference exact_alias_fallback\n";
  }
}

/// Compare the independent-column inverse with the scalar path and native math.
void inverseLuCases(Queue& queue)
{
  using namespace device_determinant;
  constexpr const char* selector = "PSIFORMER_GPU_COOPERATIVE_LU_INVERSE";

  /// Restore the caller's dispatch policy after exercising both inverse routes.
  struct RestoreSelector
  {
    const char* name;
    bool present;
    std::string value;

    explicit RestoreSelector(const char* key)
        : name(key), present(std::getenv(key) != nullptr), value(present ? std::getenv(key) : "")
    {
    }

    ~RestoreSelector()
    {
      if (present)
        setenv(name, value.c_str(), 1);
      else
        unsetenv(name);
    }
  } restore(selector);

  for (const std::size_t n : {2u, 63u, 64u, 65u, 100u, 200u, 300u})
  {
    constexpr std::size_t batch = 2, channels = 7, count = batch * channels;
    const auto elements = n * n;
    std::vector<double> input(count * elements);
    for (std::size_t m = 0; m < count; ++m)
    {
      double* a = input.data() + m * elements;
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          a[i * n + j] = i == j ? 4.0 + (i % 3) / 8.0 :
              (static_cast<int>((i * 7 + j * 3) % 9) - 4) / (16.0 * n);
      if (m == 1)
        for (std::size_t j = 0; j < n; ++j)
          std::swap(a[j], a[(n - 1) * n + j]);
      if (m == 2 || m == 3 || m == 11 || m == 13)
      {
        std::fill(a, a + elements, 0.0);
        for (std::size_t i = 0; i < n; ++i)
          a[i * n + i] = 1.0;
        if (m == 2)
        {
          a[n] = 1.0;
          if (n > 2)
            a[2 * n] = -1.0;
        }
        if (m == 3)
          a[0] = -1.0;
        if (m == 11)
          a[(n - 1) * n + n - 1] = std::ldexp(1.0, -40);
        if (m == 13)
          a[(n - 1) * n + n - 1] = 0.0;
      }
      if (m == 4)
        std::copy(a, a + n, a + (n - 1) * n);
      if (m == 5)
        std::fill(a, a + elements, 0.0);
      if (m == 6)
        a[n + 1] = std::numeric_limits<double>::quiet_NaN();
      if (m == 7)
        a[n + 1] = std::numeric_limits<double>::infinity();
      if (m == 8)
        a[n + 1] = -std::numeric_limits<double>::infinity();
      if (m == 9 || m == 10)
        for (std::size_t i = 0; i < elements; ++i)
          a[i] *= m == 9 ? 1e250 : 1e-250;
      if (m == 12)
      {
        // Normal input coefficients, finite LU, but an overflowing inverse.
        std::fill(a, a + elements, 0.0);
        const double base = std::ldexp(1.0, -830);
        for (std::size_t i = 0; i < n; ++i)
          a[i * n + i] = base;
        for (std::size_t i = 1; i < std::min(n, std::size_t(64)); ++i)
          a[i * n + i - 1] = -16 * base;
      }
    }
    auto matrices = upload(input);
    DeviceBuffer<double> scalar_lu(input.size()), parallel_lu(input.size()), scalar_inverse(input.size()),
        parallel_inverse(input.size()), solve(count * n);
    DeviceBuffer<std::size_t> scalar_perm(count * n), parallel_perm(count * n);
    DeviceBuffer<FactorizationMetadata> scalar_meta(count), parallel_meta(count);
    const auto before = deviceMemoryMetrics(0).snapshot();
    setenv(selector, "0", 1);
    checkRuntime(launchDeterminantFactorization(queue.getNative(), matrices.data(), batch, channels, n, true,
                     scalar_lu.data(), scalar_inverse.data(), scalar_perm.data(), solve.data(), scalar_meta.data()),
                 "scalar inverse reference");
    setenv(selector, "1", 1);
    checkRuntime(launchDeterminantFactorization(queue.getNative(), matrices.data(), batch, channels, n, true,
                     parallel_lu.data(), parallel_inverse.data(), parallel_perm.data(), solve.data(), parallel_meta.data()),
                 "parallel inverse");
    queue.sync();
    const auto after = deviceMemoryMetrics(0).snapshot();
    if (before.allocation_calls != after.allocation_calls || before.live_bytes != after.live_bytes)
      throw std::runtime_error("Cooperative inverse allocated hidden workspace");
    const auto old_lu = download(scalar_lu), new_lu = download(parallel_lu);
    const auto old_inverse = download(scalar_inverse), new_inverse = download(parallel_inverse);
    const auto old_perm = download(scalar_perm), new_perm = download(parallel_perm);
    const auto old_meta = download(scalar_meta), new_meta = download(parallel_meta);
    if (std::memcmp(old_lu.data(), new_lu.data(), old_lu.size() * sizeof(double)) ||
        std::memcmp(old_inverse.data(), new_inverse.data(), old_inverse.size() * sizeof(double)) || old_perm != new_perm)
      throw std::runtime_error("Cooperative inverse changed scalar LU/inverse/permutation bits");
    std::vector<double> cpu_lu(elements), cpu_inverse(elements), cpu_solve(n);
    std::vector<std::size_t> cpu_perm(n);
    for (std::size_t m = 0; m < count; ++m)
    {
      const auto& a = new_meta[m];
      const auto& b = old_meta[m];
      if (a.status != b.status || a.inverse_available != b.inverse_available ||
          std::memcmp(&a.phase, &b.phase, sizeof(double)) || std::memcmp(&a.log_abs, &b.log_abs, sizeof(double)) ||
          std::memcmp(&a.matrix_scale, &b.matrix_scale, sizeof(double)) ||
          std::memcmp(&a.minimum_scaled_pivot, &b.minimum_scaled_pivot, sizeof(double)) ||
          std::memcmp(&a.maximum_scaled_pivot, &b.maximum_scaled_pivot, sizeof(double)))
        throw std::runtime_error("Cooperative inverse changed scalar metadata bits");
      const auto cpu = factorizeScaledReal(input.data() + m * elements, n, cpu_lu.data(), cpu_perm.data(),
          true, cpu_inverse.data(), cpu_solve.data());
      if (a.status != cpu.status || a.inverse_available != cpu.inverse_available)
        throw std::runtime_error("Cooperative inverse changed native CPU admission");
      for (std::size_t i = 0; i < elements; ++i)
        near(new_inverse[m * elements + i], cpu_inverse[i], "native CPU inverse");
    }
    if (n >= 64 && (new_meta[12].status != FactorizationStatus::REGULAR || new_meta[12].inverse_available))
      throw std::runtime_error("Inverse overflow fixture lost its finite-LU premise");
    auto failed_input = input;
    failed_input[0] = std::numeric_limits<double>::quiet_NaN();
    matrices.upload(failed_input.data(), failed_input.size());
    checkRuntime(launchDeterminantFactorization(queue.getNative(), matrices.data(), batch, channels, n, true,
                     parallel_lu.data(), parallel_inverse.data(), parallel_perm.data(), solve.data(), parallel_meta.data()),
                 "inverse failure refresh");
    queue.sync();
    const auto failed_meta = download(parallel_meta);
    const auto failed_inverse = download(parallel_inverse);
    if (failed_meta[0].status != FactorizationStatus::NONFINITE_INPUT || failed_meta[0].inverse_available ||
        std::any_of(failed_inverse.begin(), failed_inverse.begin() + elements, [](double x) { return x != 0.0; }))
      throw std::runtime_error("Cooperative inverse failure retained old inverse admission");
    matrices.upload(input.data(), input.size());
    checkRuntime(launchDeterminantFactorization(queue.getNative(), matrices.data(), batch, channels, n, true,
                     parallel_lu.data(), parallel_inverse.data(), parallel_perm.data(), solve.data(), parallel_meta.data()),
                 "inverse recovery");
    queue.sync();
    const auto recovered_inverse = download(parallel_inverse);
    if (std::memcmp(old_inverse.data(), recovered_inverse.data(), old_inverse.size() * sizeof(double)))
      throw std::runtime_error("Cooperative inverse recovery changed scalar output");
    auto alias = upload(input);
    checkRuntime(launchDeterminantFactorization(queue.getNative(), alias.data(), batch, channels, n, true,
                     alias.data(), parallel_inverse.data(), parallel_perm.data(), solve.data(), parallel_meta.data()),
                 "in-place inverse fallback");
    queue.sync();
    const auto alias_lu = download(alias), alias_inverse = download(parallel_inverse);
    if (std::memcmp(old_lu.data(), alias_lu.data(), old_lu.size() * sizeof(double)) ||
        std::memcmp(old_inverse.data(), alias_inverse.data(), old_inverse.size() * sizeof(double)))
      throw std::runtime_error("Cooperative inverse changed exact in-place fallback");
    std::cout << "PASS cooperative_LU_inverse Ne=" << n << " matrices=" << count
              << " exact_GPU_LU_inverse_permutation_metadata native_CPU_inverse zero_hidden_alloc alias_fallback\n";
  }
}

/// Preserve rejected GPU scratch exactly; compare only valid CPU products numerically.
void compareOpenFeature(double actual, double expected, bool exact)
{
  if (exact)
  {
    if (std::memcmp(&actual, &expected, sizeof(double)))
      throw std::runtime_error("OPEN feature fallback/replay/padding bit mismatch");
  }
  else
    near(actual, expected, "OPEN feature CPU product");
}

/// Compare one padded two-configuration feature layout across good/failure/recovery cases.
void openFeatureCooperativeOne(Queue& queue, std::size_t ne, std::size_t nuclei_count, SpatialJetMode mode,
         std::size_t begin, std::size_t targets, bool hazards)
{
  constexpr std::size_t batch = 2;
  const std::size_t width = 4 * nuclei_count + 1, elements = ne * width;
  const std::size_t planes = mode == SpatialJetMode::FULL_VGL ? 1 + 4 * targets :
      mode == SpatialJetMode::ACTIVE ? 4 : 1;
  auto layout = makeOpenFeatureJetLayout(batch, ne, nuclei_count, ne / 2, mode,
      5, ne * 5 + 11, 7);
  layout.output = mode == SpatialJetMode::FULL_VGL ?
      makeSpatialTargetJetLayout(batch, elements, ne, begin, targets,
          elements + 13, planes * (elements + 13) + 19) :
      makeSpatialJetLayout(batch, elements, ne, mode,
          elements + 13, planes * (elements + 13) + 19);
  validateOpenFeatureJetLayout(layout);
  std::vector<double> positions(batch * layout.position_configuration_stride, 17.0);
  std::vector<double> nuclei(nuclei_count * layout.nucleus_row_stride, 23.0);
  for (std::size_t b = 0; b < batch; ++b)
    for (std::size_t e = 0; e < ne; ++e)
      for (std::size_t d = 0; d < 3; ++d)
        positions[b * layout.position_configuration_stride + 5 * e + d] = .4 + .025 * e + .1 * d + .03 * b;
  for (std::size_t n = 0; n < nuclei_count; ++n)
    for (std::size_t d = 0; d < 3; ++d)
      nuclei[7 * n + d] = -.3 - .04 * n + .07 * d;
  const auto original_positions = positions, original_nuclei = nuclei;
  auto coordinates = upload(positions), ions = upload(nuclei);
  std::vector<std::size_t> active{ne - 1, 0};
  auto active_device = upload(active);
  const std::size_t span = spatialJetSpanElements(layout.output) + 7;
  std::vector<unsigned char> logical_output(span, 0);
  for (std::size_t b = 0; b < batch; ++b)
    for (std::size_t plane = 0; plane < layout.output.plane_count; ++plane)
      for (std::size_t element = 0; element < elements; ++element)
        logical_output[layout.output.uncheckedPlaneOffset(b, plane, element)] = 1;
  std::vector<double> initial(span, 97.0), actual(span), scalar(span), cpu(span);
  auto candidate_output = upload(initial), scalar_output = upload(initial);
  DeviceBuffer<OpenSpatialStatus> candidate_status(batch), scalar_status(batch);
  std::vector<OpenSpatialStatus> actual_status(batch), old_status(batch), expected_status(batch);
  // Good/bad/good uses the same owners and padded output span throughout.
  const int variants = hazards ? 7 : 1;
  for (int variant = 0; variant < variants; ++variant)
  {
    positions = original_positions;
    nuclei = original_nuclei;
    active = {ne - 1, 0};
    if (variant == 1) active[1] = ne;
    if (variant == 2) positions[5 * 3 + 1] = std::numeric_limits<double>::quiet_NaN();
    if (variant == 3) nuclei[7 + 2] = std::numeric_limits<double>::infinity();
    if (variant == 4 || variant == 5)
      for (std::size_t d = 0; d < 3; ++d) positions[5 * (ne - 1) + d] = nuclei[d];
    // Earlier computed nonfinite must NOT mask the later pair's coalescence.
    if (variant == 5) positions[0] = 1e308;
    coordinates.upload(positions.data(), positions.size());
    ions.upload(nuclei.data(), nuclei.size());
    active_device.upload(active.data(), active.size());
    candidate_output.upload(initial.data(), initial.size());
    scalar_output.upload(initial.data(), initial.size());
    cpu = initial;
    for (std::size_t b = 0; b < batch; ++b)
      expected_status[b] = open_spatial::buildOpenFeatureJetsForConfiguration(
          layout, positions.data(), nuclei.data(), b, active[b], cpu.data());
    const auto before = deviceMemoryMetrics(0).snapshot();
    checkRuntime(launchOpenFeatureJets(queue.getNative(), layout, coordinates.data(), ions.data(),
        active_device.data(), scalar_output.data(), scalar_status.data()), "scalar OPEN feature oracle");
    checkRuntime(launchOpenFeatureJetsCooperative(queue.getNative(), layout, coordinates.data(), ions.data(),
        active_device.data(), candidate_output.data(), candidate_status.data()), "cooperative OPEN features");
    queue.sync();
    const auto after = deviceMemoryMetrics(0).snapshot();
    if (before.allocation_calls != after.allocation_calls || before.free_calls != after.free_calls ||
        before.live_bytes != after.live_bytes)
      throw std::runtime_error("OPEN feature launch changed tracked owners");
    candidate_output.download(actual.data(), actual.size());
    scalar_output.download(scalar.data(), scalar.size());
    candidate_status.download(actual_status.data(), batch);
    scalar_status.download(old_status.data(), batch);
    if (actual_status != old_status || actual_status != expected_status)
      throw std::runtime_error("OPEN feature canonical status differs");
    if (variant == 1 && actual_status[1] != OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT)
      throw std::runtime_error("Invalid-active fixture lost premise");
    if (variant == 4 && mode == SpatialJetMode::VALUE_ONLY && actual_status[0] != OpenSpatialStatus::REGULAR)
      throw std::runtime_error("VALUE feature coalescence must remain regular");
    if (variant == 5 && actual_status[0] != (mode == SpatialJetMode::VALUE_ONLY ?
        OpenSpatialStatus::NONFINITE_INPUT_OR_RESULT : OpenSpatialStatus::COALESCENCE))
      throw std::runtime_error("Feature overflow/coalescence failure precedence differs");
    for (std::size_t i = 0; i < span; ++i)
    {
      // Failed configurations replay the identical scalar device helper; compare ALL partial scratch.
      bool failed_configuration = false;
      for (std::size_t b = 0; b < batch; ++b)
        if (i >= b * layout.output.configuration_stride &&
            i < b * layout.output.configuration_stride +
                (layout.output.plane_count - 1) * layout.output.plane_stride + elements)
          failed_configuration = actual_status[b] != OpenSpatialStatus::REGULAR;
      if (!logical_output[i])
      {
        compareOpenFeature(actual[i], initial[i], true);
        compareOpenFeature(scalar[i], initial[i], true);
      }
      compareOpenFeature(actual[i], scalar[i], !logical_output[i] || ne < 64 || failed_configuration);
      // Rejected scratch is unpublished: preserve the old GPU bits, not a
      // cross-backend NaN/Inf class from already-invalid host arithmetic.
      if (!failed_configuration)
        compareOpenFeature(actual[i], cpu[i], false);
    }
  }
  std::cout << "PASS OPEN_feature_cooperative Ne=" << ne << " nuclei=" << nuclei_count
            << " B=2 mode=" << static_cast<int>(mode) << " target_begin=" << begin
            << " target_count=" << targets << " variants=" << variants
            << " all_planes_padding_status_partial_scratch_zero_tracked_alloc\n";
}

/// Bound raw execution-map coverage, including ECP VALUE tails and failure/recovery.
void openFeatureCooperativeCases(Queue& queue)
{
  openFeatureCooperativeOne(queue, 63, 3, SpatialJetMode::ACTIVE, 0, 0, false);
  openFeatureCooperativeOne(queue, 64, 5, SpatialJetMode::FULL_VGL, 63, 1, false);
  openFeatureCooperativeOne(queue, 129, 3, SpatialJetMode::FULL_VGL, 127, 2, false);
  openFeatureCooperativeOne(queue, 100, 100, SpatialJetMode::FULL_VGL, 37, 1, false);
  openFeatureCooperativeOne(queue, 64, 2, SpatialJetMode::VALUE_ONLY, 0, 0, false);
  openFeatureCooperativeOne(queue, 63, 3, SpatialJetMode::VALUE_ONLY, 0, 0, true);
  openFeatureCooperativeOne(queue, 100, 40, SpatialJetMode::VALUE_ONLY, 0, 0, true);
  openFeatureCooperativeOne(queue, 129, 3, SpatialJetMode::VALUE_ONLY, 0, 0, true);
  openFeatureCooperativeOne(queue, 64, 2, SpatialJetMode::ACTIVE, 0, 0, false);
  openFeatureCooperativeOne(queue, 64, 2, SpatialJetMode::FULL_VGL, 63, 1, true);
}

/// Validate open pair features, including zero and small-radius Taylor limits.
void features(Queue& queue)
{
  const std::vector<double> radii{0.0, 1e-8, 0.003, 0.1, 2.0, 12.0};
  std::vector<double> displacement(3 * radii.size(), 0.0);
  for (std::size_t pair = 0; pair < radii.size(); ++pair)
    displacement[3 * pair] = radii[pair];
  auto r = upload(radii);
  auto d = upload(displacement);
  DeviceBuffer<double> output(4 * radii.size());
  checkRuntime(launchPairFeatures(queue.getNative(), d.data(), nullptr, r.data(),
                                  radii.size(), false, output.data()), "pair features");
  queue.sync();
  const auto result = download(output);
  for (std::size_t pair = 0; pair < radii.size(); ++pair)
  {
    near(result[4 * pair], std::log1p(radii[pair]), "radial feature");
    near(result[4 * pair + 1], std::log1p(radii[pair]), "direction feature");
    near(result[4 * pair + 2], 0.0, "zero y feature");
    near(result[4 * pair + 3], 0.0, "zero z feature");
  }
  std::cout << "PASS open_features zero/small/regular radii\n";
}

/// Compare real-space feature gradients and per-electron trace Laplacians with FD.
void featureVGL(Queue& queue)
{
  const auto layout = makeOpenFeatureJetLayout(2, 2, 2, 1, SpatialJetMode::FULL_VGL);
  const std::vector<double> positions{0.3, 0.4, -0.2, 2.7, -0.3, 0.5,
                                      -0.2, 0.6, 0.4, 3.2, 0.5, -0.4};
  const std::vector<double> nuclei{0, 0, 0, 3.0154361724853516, 0, 0};
  auto p = upload(positions), n = upload(nuclei);
  auto active = upload(std::vector<std::size_t>(2, 0));
  DeviceBuffer<double> output(spatialJetSpanElements(layout.output));
  DeviceBuffer<OpenSpatialStatus> status(2);
  checkRuntime(launchOpenFeatureJets(queue.getNative(), layout, p.data(), n.data(),
                                     active.data(), output.data(), status.data()), "open feature VGL");
  queue.sync();
  const auto actual = download(output);
  for (auto flag : download(status))
    if (flag != OpenSpatialStatus::REGULAR)
      throw std::runtime_error("regular open feature VGL status");

  // Independent values allow finite differences to constrain plane ordering as
  // well as the kernel's radial and Cartesian derivative algebra.
  auto values = [&](const std::vector<double>& coordinates, std::size_t configuration) {
    std::vector<double> result(layout.output.element_count, 0);
    for (std::size_t electron = 0; electron < 2; ++electron)
    {
      for (std::size_t nucleus = 0; nucleus < 2; ++nucleus)
      {
        double delta[3], radius2 = 0;
        for (std::size_t dim = 0; dim < 3; ++dim)
        {
          delta[dim] = coordinates[configuration * 6 + electron * 3 + dim] - nuclei[nucleus * 3 + dim];
          radius2 += delta[dim] * delta[dim];
        }
        const double radius = std::sqrt(radius2), radial = std::log1p(radius);
        const auto offset = electron * layout.feature_width + nucleus * 4;
        result[offset] = radial;
        for (std::size_t dim = 0; dim < 3; ++dim) result[offset + 1 + dim] = delta[dim] * radial / radius;
      }
      result[electron * layout.feature_width + layout.feature_width - 1] = electron == 0 ? 1 : -1;
    }
    return result;
  };
  constexpr double h = 2e-4;
  for (std::size_t configuration = 0; configuration < 2; ++configuration)
  {
    const auto base = values(positions, configuration);
    for (std::size_t element = 0; element < base.size(); ++element)
      near(actual[layout.output.uncheckedValueOffset(configuration, element)], base[element], "VGL feature value");
    for (std::size_t electron = 0; electron < 2; ++electron)
    {
      std::vector<double> laplacian(base.size(), 0);
      for (std::size_t dim = 0; dim < 3; ++dim)
      {
        auto plus = positions, minus = positions;
        plus[configuration * 6 + electron * 3 + dim] += h;
        minus[configuration * 6 + electron * 3 + dim] -= h;
        const auto fp = values(plus, configuration), fm = values(minus, configuration);
        for (std::size_t element = 0; element < base.size(); ++element)
        {
          near(actual[layout.output.uncheckedGradientOffset(configuration, electron * 3 + dim, element)],
               (fp[element] - fm[element]) / (2 * h), "VGL feature gradient FD", 2e-7);
          laplacian[element] += (fp[element] - 2 * base[element] + fm[element]) / (h * h);
        }
      }
      for (std::size_t element = 0; element < base.size(); ++element)
        near(actual[layout.output.uncheckedLaplacianOffset(configuration, electron, element)],
             laplacian[element], "VGL feature trace FD", 2e-6);
    }
  }
  std::cout << "PASS real_space_feature_VGL coordinate_planes finite_difference\n";
}

/// Compare value, gradient and diagonal second derivative with analytic and FD oracles.
void nonlinearJets(Queue& queue)
{
  for (std::size_t count : {std::size_t(1), std::size_t(17), std::size_t(129)})
  {
    const auto values = randomValues(count, 12);
    const auto first  = randomValues(count, 13);
    const auto second = randomValues(count, 14);
    auto v = upload(values), d = upload(first), d2 = upload(second);
    DeviceBuffer<double> ov(count), od(count), od2(count);
    checkRuntime(launchTanhJets(queue.getNative(), v.data(), d.data(), d2.data(),
                                count, ov.data(), od.data(), od2.data()), "tanh jets");
    queue.sync();
    const auto actual = download(ov), actual_d = download(od), actual_d2 = download(od2);
    for (std::size_t element = 0; element < count; ++element)
    {
      const double y = std::tanh(values[element]), factor = 1 - y * y;
      near(actual[element], y, "tanh");
      near(actual_d[element], factor * first[element], "tanh first");
      near(actual_d2[element], factor * second[element] - 2 * y * factor * first[element] * first[element],
           "tanh second");
      const double h = 1e-4;
      auto shifted = [&](double t) { return std::tanh(values[element] + first[element] * t +
                                                     0.5 * second[element] * t * t); };
      near(actual_d[element], (shifted(h) - shifted(-h)) / (2 * h), "FD first", 2e-8);
      near(actual_d2[element], (shifted(h) - 2 * shifted(0) + shifted(-h)) / (h * h),
           "FD second", 2e-7);
    }
    // Reuse prepared output buffers and exercise non-multiple launch sizes.
    checkRuntime(launchResidualJets(queue.getNative(), v.data(), d.data(), d2.data(),
                                    v.data(), d.data(), d2.data(), count,
                                    ov.data(), od.data(), od2.data()), "residual jets");
    queue.sync();
    const auto sums = download(ov), sums_d = download(od), sums_d2 = download(od2);
    for (std::size_t element = 0; element < count; ++element)
    {
      near(sums[element], 2 * values[element], "residual value");
      near(sums_d[element], 2 * first[element], "residual first");
      near(sums_d2[element], 2 * second[element], "residual second");
    }
  }
  std::cout << "PASS nonlinear_spatial_jets analytic+finite_difference sizes=1,17,129\n";
}

/// Check padded row-major BLAS products and both input/parameter reverse products.
void denseAndReverse(Queue& queue, AcceleratorBlasHandle& handle)
{
  DenseForwardLayout layout{3, 4, 5, 6, 7, 8};
  const auto input = randomValues(layout.rows * layout.source_row_stride, 21);
  const auto weight = randomValues(layout.input_width * layout.weight_row_stride, 22);
  const auto seed = randomValues(layout.rows * layout.target_row_stride, 23);
  std::vector<double> expected(layout.rows * layout.target_row_stride, -999.0);
  std::vector<double> dx(layout.rows * layout.source_row_stride, -999.0);
  std::vector<double> dw(layout.input_width * layout.weight_row_stride, -999.0);
  for (std::size_t row = 0; row < layout.rows; ++row)
    for (std::size_t in = 0; in < layout.input_width; ++in)
      dx[row * layout.source_row_stride + in] = 0;
  for (std::size_t in = 0; in < layout.input_width; ++in)
    for (std::size_t out = 0; out < layout.output_width; ++out)
      dw[in * layout.weight_row_stride + out] = 0;
  auto x = upload(input), w = upload(weight), g = upload(seed), y = upload(expected);
  auto adx = upload(dx), adw = upload(dw);
  DeviceBuffer<double> bias(layout.output_width);
  const std::vector<double> zero_bias(layout.output_width, 0);
  bias.upload(zero_bias.data(), zero_bias.size());
  denseForward(handle, layout, x.data(), w.data(), y.data());
  accumulateDenseInputAdjoint(handle, layout, g.data(), w.data(), adx.data());
  accumulateDenseWeightAdjoint(handle, layout, x.data(), g.data(), adw.data());
  checkRuntime(launchDenseBiasAdjoint(queue.getNative(), layout, g.data(), bias.data()), "dense bias reverse");
  queue.sync();
  for (std::size_t row = 0; row < layout.rows; ++row)
    for (std::size_t out = 0; out < layout.output_width; ++out)
    {
      double sum = 0;
      for (std::size_t in = 0; in < layout.input_width; ++in)
      {
        sum += input[row * layout.source_row_stride + in] * weight[in * layout.weight_row_stride + out];
        dx[row * layout.source_row_stride + in] += seed[row * layout.target_row_stride + out] *
                                                  weight[in * layout.weight_row_stride + out];
        dw[in * layout.weight_row_stride + out] += input[row * layout.source_row_stride + in] *
                                                   seed[row * layout.target_row_stride + out];
      }
      expected[row * layout.target_row_stride + out] = sum;
    }
  const auto actual = download(y), actual_dx = download(adx), actual_dw = download(adw), actual_bias = download(bias);
  for (std::size_t i = 0; i < expected.size(); ++i) near(actual[i], expected[i], "dense/padding");
  for (std::size_t i = 0; i < dx.size(); ++i) near(actual_dx[i], dx[i], "dense input reverse/padding");
  for (std::size_t i = 0; i < dw.size(); ++i) near(actual_dw[i], dw[i], "dense weight reverse/padding");
  for (std::size_t out = 0; out < layout.output_width; ++out)
  {
    double sum = 0;
    for (std::size_t row = 0; row < layout.rows; ++row) sum += seed[row * layout.target_row_stride + out];
    near(actual_bias[out], sum, "dense bias reverse");
  }
  std::cout << "PASS dense_forward_input_vjp_parameter_vjp_bias padded_strides\n";
}

/// Check row/head separation, overflow-safe softmax and untouched padding.
void softmax(Queue& queue)
{
  AttentionForwardLayout layout{5, 2, 3, 6, 7, 37};
  auto logits = randomValues(74, 31);
  for (std::size_t head = 0; head < layout.heads; ++head)
    for (std::size_t row = 0; row < layout.rows; ++row)
      for (std::size_t key = 0; key < layout.rows; ++key)
        logits[layout.attentionOffset(head, row, key)] += head == 0 ? 900 : -900;
  auto expected = logits;
  for (std::size_t head = 0; head < layout.heads; ++head)
    for (std::size_t row = 0; row < layout.rows; ++row)
    {
      const auto offset = layout.attentionOffset(head, row, 0);
      const double maximum = *std::max_element(logits.begin() + offset, logits.begin() + offset + layout.rows);
      double sum = 0;
      for (std::size_t key = 0; key < layout.rows; ++key) sum += std::exp(logits[offset + key] - maximum);
      for (std::size_t key = 0; key < layout.rows; ++key)
        expected[offset + key] = std::exp(logits[offset + key] - maximum) / sum;
    }
  auto values = upload(logits);
  checkRuntime(launchAttentionSoftmax(queue.getNative(), layout, values.data()), "softmax");
  queue.sync();
  const auto actual = download(values);
  for (std::size_t i = 0; i < actual.size(); ++i) near(actual[i], expected[i], "softmax/padding");
  std::cout << "PASS attention_softmax multiple_heads padded_rows extreme_logits\n";
}

/// Exercise physical-Ne/local-target kernels and complete-output scatter on GPU.
void targetRangePrimitives(Queue& queue)
{
  constexpr std::size_t batch = 3, ne = 5, dets = 2;
  const auto mode = SpatialJetMode::FULL_VGL;
  const auto cell = periodic_geometry::makeCell(
      {{{5.1, .2, .1}, {.4, 4.8, .3}, {.2, .5, 5.3}}}, {{true, true, true}});
  std::vector<double> positions(batch * ne * 3);
  for (std::size_t i = 0; i < positions.size(); ++i) positions[i] = .21 + .037 * i;
  const std::vector<double> nuclei{.1, -.2, .3};
  auto coordinates = upload(positions), ions = upload(nuclei);
  auto active = upload(std::vector<std::size_t>(batch, 0));
  DeviceBuffer<OpenSpatialStatus> status(batch * dets * ne * ne);
  auto admit = [&](std::size_t count) {
    queue.sync();
    std::vector<OpenSpatialStatus> codes(count);
    status.download(codes.data(), count);
    for (auto code : codes)
      if (code != OpenSpatialStatus::REGULAR) throw std::runtime_error("target primitive status");
  };
  auto compare = [&](const SpatialJetLayout& full, const SpatialJetLayout& local,
                     const std::vector<double>& all, const std::vector<double>& part) {
    for (std::size_t b = 0; b < batch; ++b)
      for (std::size_t e = 0; e < full.element_count; ++e)
      {
        near(part[local.uncheckedValueOffset(b, e)], all[full.uncheckedValueOffset(b, e)], "target value");
        for (std::size_t k = 0; k < local.laplacian_lanes; ++k)
        {
          const auto physical = local.physicalTargetElectron(k);
          for (std::size_t d = 0; d < 3; ++d)
            near(part[local.uncheckedGradientOffset(b, 3 * k + d, e)],
                all[full.uncheckedGradientOffset(b, 3 * physical + d, e)], "target gradient");
          near(part[local.uncheckedLaplacianOffset(b, k, e)],
              all[full.uncheckedLaplacianOffset(b, physical, e)], "target Laplacian");
        }
      }
  };
  for (const bool periodic : {false, true})
  {
    const auto features = periodic ? makePeriodicFeatureJetLayout(batch, ne, 1, 2, mode)
                                   : makeOpenFeatureJetLayout(batch, ne, 1, 2, mode);
    const auto cusp = makeOpenCuspJetLayout(batch, ne, 2, mode);
    const auto orbitals = makeOpenOrbitalJetLayout(batch, dets, ne, 1, features.feature_width, 2, mode);
    auto weights = upload(std::vector<double>(features.feature_width * dets * ne, .17));
    auto pi = upload(std::vector<double>(dets * ne, .4)), zeta = upload(std::vector<double>(dets * ne, -.7));
    DeviceBuffer<double> full_features(spatialJetSpanElements(features.output));
    DeviceBuffer<double> full_cusp(spatialJetSpanElements(cusp.output));
    DeviceBuffer<double> full_orbitals(spatialJetSpanElements(orbitals.orbitals));
    auto run_features = [&](const OpenFeatureJetLayout& layout, double* output) {
      checkRuntime(periodic ? launchPeriodicFeatureJets(queue.getNative(), layout, cell, coordinates.data(),
          ions.data(), active.data(), output, status.data()) : launchOpenFeatureJets(queue.getNative(), layout,
          coordinates.data(), ions.data(), active.data(), output, status.data()), "target feature jets");
      admit(batch);
    };
    auto run_cusp = [&](const OpenCuspJetLayout& layout, double* output) {
      checkRuntime(periodic ? launchPeriodicCuspJets(queue.getNative(), layout, cell, coordinates.data(),
          active.data(), .8, 1.1, output, status.data()) : launchOpenCuspJets(queue.getNative(), layout,
          coordinates.data(), active.data(), .8, 1.1, output, status.data()), "target cusp jets");
      admit(batch);
    };
    auto run_orbitals = [&](const OpenOrbitalJetLayout& layout, const double* source, double* output) {
      checkRuntime(periodic ? launchPeriodicOrbitalJets(queue.getNative(), layout, cell, source, coordinates.data(),
          ions.data(), weights.data(), weights.data(), pi.data(), pi.data(), zeta.data(), zeta.data(), active.data(),
          output, status.data()) : launchOpenOrbitalJets(queue.getNative(), layout, source, coordinates.data(),
          ions.data(), weights.data(), weights.data(), pi.data(), pi.data(), zeta.data(), zeta.data(), active.data(),
          output, status.data()), "target orbital jets");
      admit(batch * dets * ne * ne);
    };
    run_features(features, full_features.data()); run_cusp(cusp, full_cusp.data());
    run_orbitals(orbitals, full_features.data(), full_orbitals.data());
    const auto all_features = download(full_features), all_cusp = download(full_cusp), all_orbitals = download(full_orbitals);
    DeviceBuffer<double> local_features(spatialJetSpanElements(withSpatialTargetRange(features.output, 0, 2)));
    DeviceBuffer<double> local_cusp(spatialJetSpanElements(withSpatialTargetRange(cusp.output, 0, 2)));
    DeviceBuffer<double> local_orbitals(spatialJetSpanElements(withSpatialTargetRange(orbitals.orbitals, 0, 2)));
    const auto destination = makeSpatialJetLayout(batch, 1, ne, mode, 2, (1 + 4 * ne) * 2 + 7);
    auto assembled = upload(std::vector<double>(spatialJetSpanElements(destination) + 1, 97.0));
    auto phases = upload(std::vector<double>(batch, 97.0)), ratios = upload(std::vector<double>(batch * ne, 97.0));
    auto local_phases = upload(std::vector<double>(batch, -1.0));
    DeviceBuffer<double> local_ratios(batch * 2);
    for (std::size_t begin = 0; begin < ne; begin += 2)
    {
      const auto count = std::min(std::size_t{2}, ne - begin);
      auto feature_tile = features; feature_tile.output = withSpatialTargetRange(features.output, begin, count);
      auto cusp_tile = cusp; cusp_tile.output = withSpatialTargetRange(cusp.output, begin, count);
      auto orbital_tile = orbitals;
      orbital_tile.features = withSpatialTargetRange(orbitals.features, begin, count);
      orbital_tile.orbitals = withSpatialTargetRange(orbitals.orbitals, begin, count);
      run_features(feature_tile, local_features.data()); run_cusp(cusp_tile, local_cusp.data());
      run_orbitals(orbital_tile, local_features.data(), local_orbitals.data());
      compare(features.output, feature_tile.output, all_features, download(local_features));
      compare(cusp.output, cusp_tile.output, all_cusp, download(local_cusp));
      compare(orbitals.orbitals, orbital_tile.orbitals, all_orbitals, download(local_orbitals));
      std::vector<double> tile_ratios(batch * count), tile_phases(batch, begin == 0 ? -1.0 : 1.0);
      for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t k = 0; k < count; ++k) tile_ratios[b * count + k] = 1.2 + .1 * (b * ne + begin + k);
      local_ratios.upload(tile_ratios.data(), tile_ratios.size());
      local_phases.upload(tile_phases.data(), tile_phases.size());
      checkRuntime(launchScatterSpatialTargets(queue.getNative(), cusp_tile.output, destination, local_phases.data(),
          local_cusp.data(), local_ratios.data(), phases.data(), assembled.data(), ratios.data()), "target root scatter");
    }
    queue.sync();
    const auto result = download(assembled), result_ratios = download(ratios), result_phases = download(phases);
    compare(cusp.output, destination, all_cusp, result);
    for (std::size_t b = 0; b < batch; ++b) near(result_phases[b], -1.0, "first-tile canonical phase");
    for (std::size_t i = 0; i < result_ratios.size(); ++i) near(result_ratios[i], 1.2 + .1 * i, "target ratio mapping");
    near(result.back(), 97.0, "target scatter trailing guard");
  }
  std::cout << "PASS target range GPU primitives: physical Ne5, B3, D2, open/skew periodic, nonzero starts/tail, scatter\n";
}

/// Compare factorization, inverse and signed determinant combinations with 2x2 algebra.
void determinants(Queue& queue)
{
  constexpr std::size_t configurations = 3, channels = 2, size = 2;
  std::vector<double> matrices(4 * configurations * channels);
  for (std::size_t term = 0; term < configurations * channels; ++term)
  {
    matrices[4 * term] = 1.2 + term * 0.1;
    matrices[4 * term + 1] = 0.2;
    matrices[4 * term + 2] = -0.3;
    matrices[4 * term + 3] = 1.8 - term * 0.05;
  }
  const std::vector<double> coefficients{1, -0.2};
  auto a = upload(matrices), c = upload(coefficients);
  DeviceBuffer<double> lu(matrices.size()), inverse(matrices.size()), solve(2 * configurations * channels);
  DeviceBuffer<std::size_t> permutations(size * configurations * channels);
  DeviceBuffer<device_determinant::FactorizationMetadata> factors(configurations * channels);
  DeviceBuffer<device_determinant::CombinationMetadata> roots(configurations);
  DeviceBuffer<double> phases(configurations * channels), logs(configurations * channels),
      scaled(configurations * channels), normalized(configurations * channels);
  checkRuntime(launchDeterminantFactorization(queue.getNative(), a.data(), configurations,
      channels, size, true, lu.data(), inverse.data(), permutations.data(), solve.data(), factors.data()), "determinants");
  checkRuntime(launchDeterminantCombination(queue.getNative(), factors.data(), c.data(), configurations,
      channels, phases.data(), logs.data(), scaled.data(), normalized.data(), roots.data()), "determinant sum");
  queue.sync();
  const auto actual = download(factors);
  const auto actual_roots = download(roots);
  const auto actual_inverse = download(inverse), weights = download(normalized);
  for (std::size_t configuration = 0; configuration < configurations; ++configuration)
  {
    double psi = 0;
    for (std::size_t channel = 0; channel < channels; ++channel)
    {
      const std::size_t term = configuration * channels + channel, offset = 4 * term;
      const double det = matrices[offset] * matrices[offset + 3] - matrices[offset + 1] * matrices[offset + 2];
      if (actual[term].status != device_determinant::FactorizationStatus::REGULAR)
        throw std::runtime_error("regular determinant status");
      near(actual[term].log_abs, std::log(std::abs(det)), "determinant log");
      near(actual_inverse[offset], matrices[offset + 3] / det, "inverse 00");
      near(actual_inverse[offset + 1], -matrices[offset + 1] / det, "inverse 01");
      near(actual_inverse[offset + 2], -matrices[offset + 2] / det, "inverse 10");
      near(actual_inverse[offset + 3], matrices[offset] / det, "inverse 11");
      psi += coefficients[channel] * det;
    }
    if (actual_roots[configuration].status != device_determinant::CombinationStatus::REGULAR)
      throw std::runtime_error("regular combination status");
    near(actual_roots[configuration].log_abs, std::log(std::abs(psi)), "combined log");
    near(weights[configuration * channels] + weights[configuration * channels + 1], 1.0, "normalized channels");
  }
  // Independent 2x2 polynomial derivatives cover both the legacy contiguous
  // ABI and separate padded configuration views, including a partial live batch.
  constexpr std::size_t electrons = 2, lanes = 6, channel_elements = 8;
  for (const bool padded : {false, true})
  {
    const std::size_t gradient_stride = padded ? 79 : lanes * channel_elements;
    const std::size_t laplacian_stride = padded ? 31 : electrons * channel_elements;
    std::vector<double> gradients(configurations * gradient_stride,
        std::numeric_limits<double>::quiet_NaN());
    std::vector<double> laplacians(configurations * laplacian_stride,
        std::numeric_limits<double>::quiet_NaN());
    for (std::size_t configuration = 0; configuration < configurations; ++configuration)
    {
      for (std::size_t lane = 0; lane < lanes; ++lane)
        for (std::size_t element = 0; element < channel_elements; ++element)
          gradients[configuration * gradient_stride + lane * channel_elements + element] =
              0.017 * (1 + configuration) * (1 + lane) * (static_cast<int>(element % 5) - 2);
      for (std::size_t electron = 0; electron < electrons; ++electron)
        for (std::size_t element = 0; element < channel_elements; ++element)
          laplacians[configuration * laplacian_stride + electron * channel_elements + element] =
              0.011 * (1 + configuration) * (1 + electron) * (static_cast<int>(element % 3) - 1);
    }
    auto gradient = upload(gradients), laplacian = upload(laplacians);
    DeviceBuffer<double> product(configurations * size * size);
    DeviceBuffer<double> log_gradient(configurations * lanes), lap_ratio(configurations * electrons),
        lap_log(configurations * electrons);
    DeviceBuffer<device_determinant::DerivativeStatus> status(configurations);
    for (const std::size_t live : {configurations, std::size_t{1}, std::size_t{2}, configurations})
    {
      const std::vector<double> gradient_marker(configurations * lanes, 19.0);
      const std::vector<double> laplacian_marker(configurations * electrons, 19.0);
      log_gradient.upload(gradient_marker.data(), gradient_marker.size());
      lap_ratio.upload(laplacian_marker.data(), laplacian_marker.size());
      lap_log.upload(laplacian_marker.data(), laplacian_marker.size());
      device_determinant::SpatialLayout layout{live, channels, size, lanes, electrons,
          padded ? gradient_stride : 0, padded ? laplacian_stride : 0};
      checkRuntime(launchDeterminantSpatialTraces(queue.getNative(), layout, factors.data(), roots.data(),
          normalized.data(), inverse.data(), gradient.data(), laplacian.data(), product.data(),
          log_gradient.data(), lap_ratio.data(), lap_log.data(), status.data()), "strided determinant traces");
      queue.sync();
      const auto result_gradient = download(log_gradient), result_ratio = download(lap_ratio),
          result_log = download(lap_log);
      const auto result_status = download(status);
      for (std::size_t configuration = 0; configuration < configurations; ++configuration)
      {
        if (configuration >= live)
        {
          for (std::size_t lane = 0; lane < lanes; ++lane)
            near(result_gradient[configuration * lanes + lane], 19.0, "spatial gradient inactive tail");
          for (std::size_t electron = 0; electron < electrons; ++electron)
          {
            near(result_ratio[configuration * electrons + electron], 19.0, "spatial ratio inactive tail");
            near(result_log[configuration * electrons + electron], 19.0, "spatial log inactive tail");
          }
          continue;
        }
        if (result_status[configuration] != device_determinant::DerivativeStatus::AVAILABLE)
          throw std::runtime_error("strided determinant trace availability");
        double psi = 0.0;
        for (std::size_t channel = 0; channel < channels; ++channel)
        {
          const double* matrix = matrices.data() + configuration * channel_elements + 4 * channel;
          psi += coefficients[channel] * (matrix[0] * matrix[3] - matrix[1] * matrix[2]);
        }
        for (std::size_t electron = 0; electron < electrons; ++electron)
        {
          double psi_laplacian = 0.0, gradient_square = 0.0;
          for (std::size_t channel = 0; channel < channels; ++channel)
          {
            const double* matrix = matrices.data() + configuration * channel_elements + 4 * channel;
            const double* lap = laplacians.data() + configuration * laplacian_stride +
                electron * channel_elements + 4 * channel;
            double channel_laplacian = lap[0] * matrix[3] + matrix[0] * lap[3] -
                lap[1] * matrix[2] - matrix[1] * lap[2];
            for (std::size_t dimension = 0; dimension < 3; ++dimension)
            {
              const double* g = gradients.data() + configuration * gradient_stride +
                  (3 * electron + dimension) * channel_elements + 4 * channel;
              channel_laplacian += 2 * (g[0] * g[3] - g[1] * g[2]);
            }
            psi_laplacian += coefficients[channel] * channel_laplacian;
          }
          for (std::size_t dimension = 0; dimension < 3; ++dimension)
          {
            const auto lane = 3 * electron + dimension;
            double psi_gradient = 0.0;
            for (std::size_t channel = 0; channel < channels; ++channel)
            {
              const double* matrix = matrices.data() + configuration * channel_elements + 4 * channel;
              const double* g = gradients.data() + configuration * gradient_stride + lane * channel_elements + 4 * channel;
              psi_gradient += coefficients[channel] *
                  (g[0] * matrix[3] + matrix[0] * g[3] - g[1] * matrix[2] - matrix[1] * g[2]);
            }
            const double expected_gradient = psi_gradient / psi;
            near(result_gradient[configuration * lanes + lane], expected_gradient, "strided determinant log gradient");
            gradient_square += expected_gradient * expected_gradient;
          }
          near(result_ratio[configuration * electrons + electron], psi_laplacian / psi, "strided determinant ratio");
          near(result_log[configuration * electrons + electron], psi_laplacian / psi - gradient_square,
              "strided determinant log Laplacian");
        }
      }
    }
  }
  std::cout << "PASS determinant_spatial_views contiguous padded_independent_strides live_tails\n";

  // Reuse allocations to test singular and nonfinite statuses separately.
  matrices[0] = 1; matrices[1] = 2; matrices[2] = 2; matrices[3] = 4;
  matrices[4] = std::numeric_limits<double>::quiet_NaN();
  a.upload(matrices.data(), matrices.size());
  checkRuntime(launchDeterminantFactorization(queue.getNative(), a.data(), configurations,
      channels, size, true, lu.data(), inverse.data(), permutations.data(), solve.data(), factors.data()), "bad determinant statuses");
  queue.sync();
  const auto bad = download(factors);
  if (bad[0].status != device_determinant::FactorizationStatus::SINGULAR ||
      bad[1].status != device_determinant::FactorizationStatus::NONFINITE_INPUT)
    throw std::runtime_error("determinant bad-input status");
  std::cout << "PASS determinants inverse signed_sum singular_nonfinite\n";
}

/// Prove request-local exact reuse without counters or additional production owners.
void determinantTargetReuse(Queue& queue)
{
  constexpr std::size_t batch = 3, channels = 2, size = 2, stride = 11, count = batch * channels;
  std::vector<double> source(batch * stride, std::numeric_limits<double>::quiet_NaN());
  for (std::size_t matrix = 0; matrix < count; ++matrix)
  {
    const auto origin = (matrix / channels) * stride + (matrix % channels) * 4;
    source[origin] = 1.2 + matrix * 0.1;
    source[origin + 1] = 0.0;
    source[origin + 2] = -0.3;
    source[origin + 3] = 1.8;
  }
  auto input = upload(source);
  DeviceBuffer<double> packed(4 * count), lu(4 * count), inverse(4 * count), solve(size * count);
  DeviceBuffer<std::size_t> permutation(size * count);
  DeviceBuffer<device_determinant::FactorizationMetadata> metadata(count);
  const std::vector<double> sentinel(4 * count, 19.0);
  auto run = [&](bool reuse) {
    input.upload(source.data(), source.size());
    checkRuntime(launchDeterminantTargetFactorization(queue.getNative(), input.data(), batch,
        channels, size, stride, reuse, packed.data(), lu.data(), inverse.data(), permutation.data(),
        solve.data(), metadata.data()), "target determinant reuse");
    queue.sync();
  };
  run(false);
  const auto first_inverse = download(inverse);
  lu.upload(sentinel.data(), sentinel.size());
  run(true);
  if (download(lu) != sentinel || download(inverse) != first_inverse)
    throw std::runtime_error("identical target bases did not preserve factors/inverses");

  // Both a one-bit numerical change and a signed-zero bit change must refactor.
  source[0] = std::nextafter(source[0], 2.0);
  source[5] = -0.0;
  run(true);
  const auto changed_lu = download(lu);
  if (changed_lu[0] == 19.0 || changed_lu[4] == 19.0)
    throw std::runtime_error("changed target base reused old factorization");
  for (std::size_t element = 8; element < changed_lu.size(); ++element)
    near(changed_lu[element], 19.0, "unchanged channel reuse sentinel");
  const auto target_inverse = download(inverse);
  const auto target_metadata = download(metadata);
  checkRuntime(launchDeterminantFactorization(queue.getNative(), packed.data(), batch, channels,
      size, true, lu.data(), inverse.data(), permutation.data(), solve.data(), metadata.data()),
      "independent legacy target factorization");
  queue.sync();
  const auto reference_inverse = download(inverse);
  const auto reference_metadata = download(metadata);
  for (std::size_t element = 0; element < target_inverse.size(); ++element)
    near(target_inverse[element], reference_inverse[element], "target fallback inverse");
  for (std::size_t matrix = 0; matrix < count; ++matrix)
  {
    near(target_metadata[matrix].log_abs, reference_metadata[matrix].log_abs, "target fallback log");
    if (target_metadata[matrix].status != reference_metadata[matrix].status ||
        !target_metadata[matrix].inverse_available)
      throw std::runtime_error("target fallback metadata");
  }

  // Identical data cannot reuse an unsuccessful/missing-inverse receipt.
  auto invalid = reference_metadata;
  invalid[0].inverse_available = false;
  invalid[1].status = device_determinant::FactorizationStatus::SINGULAR;
  metadata.upload(invalid.data(), invalid.size());
  lu.upload(sentinel.data(), sentinel.size());
  run(true);
  const auto repaired = download(lu);
  if (repaired[0] == 19.0 || repaired[4] == 19.0)
    throw std::runtime_error("invalid target receipt reused");
  source[0] = std::numeric_limits<double>::quiet_NaN();
  run(true);
  lu.upload(sentinel.data(), sentinel.size());
  run(true);
  if (download(lu)[0] != 0.0 ||
      download(metadata)[0].status != device_determinant::FactorizationStatus::NONFINITE_INPUT)
    throw std::runtime_error("nonfinite target receipt reused");
  source[0] = 1.2;
  run(true);
  if (!download(metadata)[0].inverse_available)
    throw std::runtime_error("target receipt recovery");
  source[0] = 1; source[1] = 2; source[2] = 2; source[3] = 4;
  run(true);
  if (download(metadata)[0].status != device_determinant::FactorizationStatus::SINGULAR ||
      download(metadata)[0].inverse_available)
    throw std::runtime_error("singular target receipt");
  source[0] = 1.2; source[1] = 0.0; source[2] = -0.3; source[3] = 1.8;
  run(true);
  near(download(inverse)[0], first_inverse[0], "singular target recovery inverse");
  lu.upload(sentinel.data(), sentinel.size());
  run(false);
  for (const auto element : download(lu))
    if (element == 19.0)
      throw std::runtime_error("first target tile failed to revoke reuse");
  std::cout << "PASS target_factor_reuse exact_bits signed_zero one_bit invalid_recovery first_tile\n";
}

/// Exercise bounded response products, finite diagnostics and ordered ECP folds.
void boundedResponses(Queue& queue)
{
  constexpr std::size_t count = 17;
  const auto response = randomValues(count, 41), direction = randomValues(count, 42);
  auto r = upload(response), d = upload(direction);
  ResponseDotLayout dot{count, 4, 5, 0, 1, 0, 1};
  DeviceBuffer<double> partials(5), result(1);
  DeviceBuffer<DerivativeFiniteStatus> partial_status(5), status(1);
  checkRuntime(launchResponseDotDirection(queue.getNative(), dot, r.data(), d.data(),
      partials.data(), partial_status.data(), result.data(), status.data()), "response dot");
  queue.sync();
  double expected = 0;
  for (std::size_t parameter = 0; parameter < count; ++parameter) expected += response[parameter] * direction[parameter];
  near(download(result)[0], expected, "response direction product");
  if (download(status)[0].nonfinite_count != 0) throw std::runtime_error("finite response dot status");

  const std::vector<double> bare{0.2, -0.1, 0.4}, ratios{1.1, 0.2, 0.8, -0.1, 1.3, 0.4};
  const std::vector<double> coefficients{0.7, -0.2, -0.3, 0.1};
  auto b = upload(bare), ratio = upload(ratios), coefficient = upload(coefficients);
  auto sums = upload(std::vector<double>(2, 0)), accumulators = upload(std::vector<double>(4 * count, 0));
  ECPWeightedResponseLayout layout{count, 2, 1, 0, 3, 0, 0, 1, 1, 2 * count};
  for (std::size_t point = 0; point < 3; ++point)
  {
    layout.point_index = point;
    checkRuntime(launchECPWeightedResponse(queue.getNative(), layout, b.data(), ratio.data(),
        coefficient.data(), r.data(), sums.data(), accumulators.data()), "ECP response fold");
  }
  queue.sync();
  std::complex<double> weight{0, 0};
  for (std::size_t point = 0; point < 3; ++point)
    weight += bare[point] * std::complex<double>(ratios[2 * point], ratios[2 * point + 1]);
  const auto actual_sums = download(sums), actual_accumulator = download(accumulators);
  near(actual_sums[0], weight.real(), "ECP complete weight real");
  near(actual_sums[1], weight.imag(), "ECP complete weight imag");
  for (std::size_t channel = 0; channel < 2; ++channel)
    for (std::size_t parameter = 0; parameter < count; ++parameter)
    {
      const auto expected_value = std::complex<double>(coefficients[2 * channel], coefficients[2 * channel + 1]) *
                                  weight * response[parameter];
      near(actual_accumulator[2 * (channel * count + parameter)], expected_value.real(), "ECP response real");
      near(actual_accumulator[2 * (channel * count + parameter) + 1], expected_value.imag(), "ECP response imag");
    }
  auto nonfinite = response;
  nonfinite[7] = std::numeric_limits<double>::quiet_NaN();
  r.upload(nonfinite.data(), count);
  DerivativeFiniteScanLayout scan{count, 4, 5, 0, 1};
  checkRuntime(launchDerivativeFiniteScan(queue.getNative(), scan, r.data(), partial_status.data()), "finite scan");
  queue.sync();
  const auto flags = download(partial_status);
  std::size_t nonfinite_count = 0;
  for (const auto& flag : flags) nonfinite_count += flag.nonfinite_count;
  if (nonfinite_count != 1 || flags[1].first_nonfinite != 7)
    throw std::runtime_error("finite scan must locate exactly one bad element");
  // Composed score admission must detect every nonfinite trunk contribution,
  // opposite infinities and finite-input overflow using the actual sum kernel.
  const double infinity = std::numeric_limits<double>::infinity();
  const double maximum = std::numeric_limits<double>::max();
  const std::vector<double> trunk{1.0, infinity, -infinity, infinity,
      -infinity, std::numeric_limits<double>::quiet_NaN(), maximum};
  const std::vector<double> head{2.0, 1.0, -1.0, -infinity, infinity, 3.0, maximum};
  auto device_trunk = upload(trunk), device_head = upload(head);
  DeviceBuffer<double> composed(trunk.size());
  DeviceBuffer<DerivativeFiniteStatus> composed_status(trunk.size());
  DerivativeFiniteScanLayout composed_scan{trunk.size(), 1, trunk.size(), 0, 1};
  checkRuntime(launchAddParameterResponses(queue.getNative(), trunk.size(),
      device_trunk.data(), device_head.data(), composed.data()), "composed response sum");
  checkRuntime(launchDerivativeFiniteScan(queue.getNative(), composed_scan, composed.data(),
      composed_status.data()), "composed response finite scan");
  queue.sync();
  const auto composed_flags = download(composed_status);
  near(download(composed)[0], 3.0, "finite composed response");
  for (std::size_t element = 0; element < trunk.size(); ++element)
    if (composed_flags[element].nonfinite_count != (element == 0 ? 0 : 1))
      throw std::runtime_error("composed response scan lost nonfinite trunk or sum overflow");

  // Reusing the exact status owner for a valid sum must clear prior failures.
  std::vector<double> valid(trunk.size(), 2.0);
  device_trunk.upload(valid.data(), valid.size());
  device_head.upload(valid.data(), valid.size());
  checkRuntime(launchAddParameterResponses(queue.getNative(), valid.size(),
      device_trunk.data(), device_head.data(), composed.data()), "valid response recomposition");
  checkRuntime(launchDerivativeFiniteScan(queue.getNative(), composed_scan, composed.data(),
      composed_status.data()), "valid composed response finite scan");
  queue.sync();
  for (const auto& flag : download(composed_status))
    if (flag.nonfinite_count != 0)
      throw std::runtime_error("composed response scan retained stale failure status");
  std::cout << "PASS bounded_response_dot_finite_scan_ecp_contraction composed_score_admission_recovery\n";
}
/// Check output-owned batched score leaves against independent scalar formulas.
void batchedScoreLeaves(Queue& queue)
{
  constexpr std::size_t batch = 3;
  const std::vector<double> weights{1.25, -0.5, 0.0};
  auto device_weights = upload(weights);

  // Deliberate stride padding and nonzero destinations catch indexing and += errors.
  std::vector<double> distances(batch * 8, 99.0), adjoints(batch * 8, 88.0);
  const std::vector<double> pi{0.7, -0.4, 77.0, 0.3, 0.9, 77.0};
  const std::vector<double> zeta{0.8, -1.1, 77.0, -0.6, 1.4, 77.0};
  std::vector<double> expected(20, 0.125);
  for (std::size_t configuration = 0; configuration < batch; ++configuration)
    for (std::size_t electron = 0; electron < 2; ++electron)
    {
      for (std::size_t nucleus = 0; nucleus < 2; ++nucleus)
        distances[configuration * 8 + electron * 3 + nucleus] =
            0.2 + 0.3 * configuration + 0.1 * electron + 0.15 * nucleus;
      for (std::size_t orbital = 0; orbital < 2; ++orbital)
        adjoints[configuration * 8 + electron * 3 + orbital] =
            weights[configuration] * (0.4 + 0.1 * electron - 0.2 * orbital);
    }
  for (std::size_t orbital = 0; orbital < 2; ++orbital)
    for (std::size_t nucleus = 0; nucleus < 2; ++nucleus)
    {
      const auto parameter = orbital * 3 + nucleus;
      for (std::size_t configuration = 0; configuration < batch; ++configuration)
        for (std::size_t electron = 0; electron < 2; ++electron)
        {
          const double radius = distances[configuration * 8 + electron * 3 + nucleus];
          const double upstream = adjoints[configuration * 8 + electron * 3 + orbital];
          const double exponential = std::exp(-std::abs(zeta[parameter]) * radius);
          expected[2 + parameter] += upstream * exponential;
          expected[12 + parameter] -= upstream * pi[parameter] * radius *
              (zeta[parameter] > 0 ? 1.0 : -1.0) * exponential;
        }
    }
  auto radii = upload(distances), upstream = upload(adjoints);
  auto p = upload(pi), z = upload(zeta), response = upload(std::vector<double>(20, 0.125));
  const EnvelopeParameterReverseLayout envelope{batch, 1, 2, 2, 0, 2,
      3, 6, 8, 3, 8, 3, 6, 2, 12};
  checkRuntime(launchEnvelopeParameterReverse(queue.getNative(), envelope, radii.data(),
      p.data(), z.data(), upstream.data(), response.data()), "batched envelope scores");
  queue.sync();
  const auto actual = download(response);
  for (std::size_t index = 0; index < expected.size(); ++index)
    near(actual[index], expected[index], "batched envelope response/padding");

  const std::vector<double> pair_distances{0.2, 0.7, 1.1, 99.0,
      0.3, 0.8, 1.2, 99.0, 0.4, 0.9, 1.3, 99.0};
  const std::vector<std::uint8_t> same_spin{1, 0, 0};
  auto pair_radii = upload(pair_distances);
  auto spin = upload(same_spin);
  auto cusp_response = upload(std::vector<double>(5, 0.25));
  std::vector<double> cusp_expected(5, 0.25);
  for (std::size_t configuration = 0; configuration < batch; ++configuration)
    for (std::size_t pair = 0; pair < 3; ++pair)
    {
      const bool same = same_spin[pair] != 0;
      const double alpha = same ? 0.8 : 1.1;
      const double factor = same ? 0.25 : 0.5;
      const double radius = pair_distances[configuration * 4 + pair];
      cusp_expected[same ? 1 : 3] -= weights[configuration] * factor * alpha *
          (alpha + 2 * radius) / ((alpha + radius) * (alpha + radius));
    }
  CuspAlphaReverseLayout cusp{batch, 3, 4, 1, 3, true};
  checkRuntime(launchCuspAlphaReverse(queue.getNative(), cusp, pair_radii.data(), spin.data(),
      0.8, 1.1, cusp_response.data(), device_weights.data()), "weighted batched cusp scores");
  queue.sync();
  const auto actual_cusp = download(cusp_response);
  for (std::size_t index = 0; index < cusp_expected.size(); ++index)
    near(actual_cusp[index], cusp_expected[index], "weighted cusp response/padding");

  // A missing same-spin parameter must not write its offset, even with same-spin pairs.
  cusp.has_same_alpha = false;
  auto opposite_only = upload(std::vector<double>(5, 0.25));
  checkRuntime(launchCuspAlphaReverse(queue.getNative(), cusp, pair_radii.data(), spin.data(),
      0.8, 1.1, opposite_only.data(), device_weights.data()), "opposite-only cusp scores");
  queue.sync();
  const auto opposite_result = download(opposite_only);
  near(opposite_result[1], 0.25, "absent same-spin parameter unchanged");
  near(opposite_result[3], cusp_expected[3], "opposite-only weighted response");
  std::cout << "PASS bounded_batched_envelope_and_weighted_cusp_scores\n";
}

} // namespace

/// Run host-only invariants or require an actual device for all numerical tests.
int main(int argc, char** argv)
{
  try
  {
    hostChecks();
    valueLuHostChecks();
    if (argc == 2 && std::string(argv[1]) == "--host-check")
    {
      std::cout << "PASS host-only ownership checks; GPU numerical tests NOT_RUN\n";
      return 0;
    }
    const bool vectors_only = argc == 2 && std::string(argv[1]) == "--real-vectors-only";
    const bool value_lu_only = argc == 2 && std::string(argv[1]) == "--value-lu-only";
    const bool inverse_lu_only = argc == 2 && std::string(argv[1]) == "--inverse-lu-only";
    const bool value_scale_only = argc == 2 && std::string(argv[1]) == "--value-scale-only";
    const bool open_feature_only = argc == 2 && std::string(argv[1]) == "--open-feature-only";
    if (argc != 1 && !vectors_only && !value_lu_only && !inverse_lu_only && !open_feature_only && !value_scale_only)
      throw std::invalid_argument("Usage: psiformer_device_runtime_validation [--host-check|--real-vectors-only|--value-lu-only|--inverse-lu-only|--open-feature-only|--value-scale-only]");
    int count = 0;
#ifdef QMC_CUDA2HIP
    const auto discovery = hipGetDeviceCount(&count);
    const char* backend = "HIP";
#else
    const auto discovery = cudaGetDeviceCount(&count);
    const char* backend = "CUDA";
#endif
    if (discovery != success || count == 0)
    {
      std::cerr << "NOT_RUN: no usable " << backend << " device; exit 77\n";
      return 77;
    }
    std::cout << "Actual-device FP64 validation backend=" << backend << " visible_devices=" << count << "\n";
    Queue queue;
    if (value_scale_only)
    {
      validateCooperativeValueScale(queue);
      return 0;
    }
    if (inverse_lu_only)
    {
      inverseLuCases(queue);
      return 0;
    }
    if (open_feature_only)
    {
      openFeatureCooperativeCases(queue);
      return 0;
    }
    if (value_lu_only)
    {
      valueLuCases(queue);
      return 0;
    }
    if (vectors_only)
    {
      validateDerivativeRealVectors(queue);
      return 0;
    }
    AcceleratorBlasHandle handle(queue);
    features(queue);
    featureVGL(queue);
    openFeatureCooperativeCases(queue);
    validateCooperativeValueCusp(queue);
    targetRangePrimitives(queue);
    nonlinearJets(queue);
    denseAndReverse(queue, handle);
    softmax(queue);
    validateAttentionJetTiles(queue);
    determinants(queue);
    valueLuCases(queue);
    inverseLuCases(queue);
    validateCooperativeValueScale(queue);
    determinantTargetReuse(queue);
    boundedResponses(queue);
    batchedScoreLeaves(queue);
    validateDerivativeRealVectors(queue);
    queue.sync();
    std::cout << "PASS all implemented device component cases; full-model dispatch NOT_VALIDATED\n";
    return 0;
  }
  catch (const std::exception& error)
  {
    std::cerr << "FAIL: " << error.what() << "\n";
    return 1;
  }
}
