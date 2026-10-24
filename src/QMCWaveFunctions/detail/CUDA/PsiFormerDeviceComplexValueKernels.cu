//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include "PsiFormerDeviceComplexValueKernels.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOpenOrbital.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cstdlib>
#include <cstring>

// Existing mapping covers complex storage, but not these arithmetic names.
// Select direct vendor operations locally; do not extend the global CUDA shim.
#ifdef QMC_CUDA2HIP
#define QMC_PF_COMPLEX_ABS hipCabs
#define QMC_PF_COMPLEX_MUL hipCmul
#define QMC_PF_COMPLEX_DIV hipCdiv
#define QMC_PF_COMPLEX_SUB hipCsub
#else
#define QMC_PF_COMPLEX_ABS cuCabs
#define QMC_PF_COMPLEX_MUL cuCmul
#define QMC_PF_COMPLEX_DIV cuCdiv
#define QMC_PF_COMPLEX_SUB cuCsub
#endif

namespace qmcplusplus::psiformer::device
{
namespace
{
/// Preserve the established binary64 node sentinel under accelerator finite-math flags.
__device__ double nodeLog()
{
  union Binary64 { std::uint64_t integer; double floating; } encoded{UINT64_C(0xfff0000000000000)};
  return encoded.floating;
}

/// Reject either nonfinite component using the shared release-safe classifier.
__device__ bool finite(cuDoubleComplex z)
{ return device_math::isFiniteBinary64(cuCreal(z)) && device_math::isFiniteBinary64(cuCimag(z)); }

/// Record first numerical failure; caller checked the index encoding before allocation.
__device__ void fail(std::uint64_t* status, std::size_t index)
{ atomicMin(reinterpret_cast<unsigned long long*>(status), static_cast<unsigned long long>((std::uint64_t(index) << 8) | 1)); }

/// Compute only real E using the unchanged shared radial/envelope arithmetic.
__global__ void envelopes(OpenOrbitalJetLayout layout, periodic_geometry::Cell cell,
    const double* positions, const double* nuclei, const double* pi_up, const double* pi_down,
    const double* zeta_up, const double* zeta_down, double* output, std::uint64_t* status)
{
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto n = layout.features.electron_count, per = layout.determinant_count * n * n;
  if (index >= layout.features.configuration_count * per)
    return;
  const auto local = index % per;
  if (open_orbital::buildElement(layout, nullptr, positions, nuclei, nullptr, nullptr,
      pi_up, pi_down, zeta_up, zeta_down, index / per, 0, local / (n * n),
      (local / n) % n, local % n, output, nullptr, nullptr, &cell, nullptr, true) != OpenSpatialStatus::REGULAR)
    fail(status, index);
}

/// Recover the base from the first tile without a separate coordinate upload or host fence.
__global__ void envelopeAnchor(OpenOrbitalJetLayout layout, periodic_geometry::Cell cell,
    const double* positions, const double* nuclei, const double* pi_up, const double* pi_down,
    const double* zeta_up, const double* zeta_down, std::size_t moved_row,
    double base_x, double base_y, double base_z, double* output, std::uint64_t* status)
{
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto n = layout.features.electron_count;
  if (index >= layout.determinant_count * n * n)
    return;
  const auto row = (index / n) % n;
  const double base_position[3]{base_x, base_y, base_z};
  if (row == moved_row)
  {
    // VALUE-only buildElement reads exactly this row; no derivative or other electron coordinate is accessed.
    layout.position_row_stride = 0;
    positions = base_position;
  }
  if (open_orbital::buildElement(layout, nullptr, positions, nuclei, nullptr, nullptr,
      pi_up, pi_down, zeta_up, zeta_down, 0, 0, index / (n * n), row, index % n,
      output, nullptr, nullptr, &cell, nullptr, true) != OpenSpatialStatus::REGULAR)
    fail(status, index);
}

/// A failed anchor is private: canonical recomputation preserves the original point's failure order.
__global__ void virtualEnvelopes(OpenOrbitalJetLayout layout, periodic_geometry::Cell cell,
    const double* positions, const double* nuclei, const double* pi_up, const double* pi_down,
    const double* zeta_up, const double* zeta_down, const double* anchor,
    const std::size_t* moved_rows, const std::uint64_t* anchor_status,
    double* output, std::uint64_t* status)
{
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto n = layout.features.electron_count, per = layout.determinant_count * n * n;
  if (index >= layout.features.configuration_count * per)
    return;
  const auto local = index % per, configuration = index / per;
  const auto row = (local / n) % n;
  const auto moved = moved_rows[configuration];
  if (moved > n)
  {
    fail(status, index);
    return;
  }
  if (*anchor_status == UINT64_MAX && moved < n && row != moved)
  {
    const auto value = anchor[local];
    output[index] = value;
    if (!device_math::isFiniteBinary64(value))
      fail(status, index);
  }
  else if (open_orbital::buildElement(layout, nullptr, positions, nuclei, nullptr, nullptr,
      pi_up, pi_down, zeta_up, zeta_down, configuration, 0, local / (n * n),
      row, local % n, output, nullptr, nullptr, &cell, nullptr, true) != OpenSpatialStatus::REGULAR)
    fail(status, index);
}

/// Produce the complex head without aliasing projection planes or LU storage.
__global__ void pack(std::size_t count, const double* envelope, const double* real,
    const double* imag, cuDoubleComplex* matrix, std::uint64_t* status)
{
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count)
    return;
  const auto value = make_cuDoubleComplex(envelope[index] * real[index], envelope[index] * imag[index]);
  matrix[index] = value;
  if (!finite(value))
    fail(status, index);
}

/// Solve scaled LU columns using its final row permutation, then restore A's scale.
__device__ bool inverseFromLu(const cuDoubleComplex* a, std::size_t n, double scale,
    const std::size_t* permutation, cuDoubleComplex* inverse, cuDoubleComplex* solve)
{
  for (std::size_t column = 0; column < n; ++column)
  {
    for (std::size_t i = 0; i < n; ++i)
    {
      auto value = make_cuDoubleComplex(permutation[i] == column ? 1 : 0, 0);
      for (std::size_t j = 0; j < i; ++j)
        value = QMC_PF_COMPLEX_SUB(value, QMC_PF_COMPLEX_MUL(a[i * n + j], solve[j]));
      if (!finite(value))
        return false;
      solve[i] = value;
    }
    for (std::size_t i = n; i-- > 0;)
    {
      auto value = solve[i];
      for (std::size_t j = i + 1; j < n; ++j)
        value = QMC_PF_COMPLEX_SUB(value, QMC_PF_COMPLEX_MUL(a[i * n + j], inverse[j * n + column]));
      inverse[i * n + column] = QMC_PF_COMPLEX_DIV(value, a[i * n + i]);
      if (!finite(inverse[i * n + column]))
        return false;
    }
  }
  for (std::size_t i = 0; i < n * n; ++i)
  {
    inverse[i] = make_cuDoubleComplex(cuCreal(inverse[i]) / scale, cuCimag(inverse[i]) / scale);
    if (!finite(inverse[i]))
      return false;
  }
  return true;
}

/// Serial-per-channel scaled partial-pivot LU; zero pivots are singular, not errors.
template<bool CollectPivots, bool EmitScale = false, bool ReuseRaw = false, bool FailedOnly = false>
__global__ void factor(std::size_t channels, std::size_t n, cuDoubleComplex* matrices,
    ComplexValueChannel* metadata, std::uint64_t* status,
    cuDoubleComplex* inverse, std::size_t* permutation, cuDoubleComplex* solve, double* minimum_scaled_pivots,
    cuDoubleComplex* canonical_raw = nullptr, bool reuse_previous = false, std::uint8_t* decisions = nullptr)
{
  const auto channel = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= channels)
    return;
  if constexpr (FailedOnly)
    if (metadata[channel].status == 0)
      return;
  auto* a = matrices + channel * n * n;
  if constexpr (ReuseRaw)
  {
    static_assert(!CollectPivots && !EmitScale);
    auto* canonical = canonical_raw + channel * n * n;
    bool identical = reuse_previous && metadata[channel].status == 0;
    for (std::size_t i = 0; identical && i < n * n; ++i)
      identical = finite(a[i]) && finite(inverse[channel * n * n + i]) &&
          __double_as_longlong(cuCreal(a[i])) == __double_as_longlong(cuCreal(canonical[i])) &&
          __double_as_longlong(cuCimag(a[i])) == __double_as_longlong(cuCimag(canonical[i]));
    if (decisions)
      decisions[channel] = identical ? 1 : (reuse_previous ? 2 : 0);
    // Do not reset accepted channels, permutations or inverse before this guard.
    if (identical)
      return;
    for (std::size_t i = 0; i < n * n; ++i)
      canonical[i] = a[i];
  }
  auto* perm = permutation ? permutation + channel * n : nullptr;
  if (perm)
    for (std::size_t i = 0; i < n; ++i)
      perm[i] = i;
  metadata[channel] = {make_cuDoubleComplex(0, 0), nodeLog(), 1};
  if constexpr (CollectPivots)
    minimum_scaled_pivots[channel] = 0;
  double minimum_pivot = INFINITY;
  double scale = 0;
  for (std::size_t i = 0; i < n * n; ++i)
  {
    if (!finite(a[i]))
    {
      fail(status, channel);
      return;
    }
    scale = fmax(scale, fmax(fabs(cuCreal(a[i])), fabs(cuCimag(a[i]))));
  }
  if (scale == 0)
    return;
  for (std::size_t i = 0; i < n * n; ++i)
    a[i] = make_cuDoubleComplex(cuCreal(a[i]) / scale, cuCimag(a[i]) / scale);
  auto phase = make_cuDoubleComplex(1, 0);
  double log_abs = double(n) * log(scale);

  // Pivot by robust vendor magnitude, retaining determinant phase through row swaps.
  for (std::size_t k = 0; k < n; ++k)
  {
    auto pivot = k;
    for (std::size_t i = k + 1; i < n; ++i)
      if (QMC_PF_COMPLEX_ABS(a[i * n + k]) > QMC_PF_COMPLEX_ABS(a[pivot * n + k]))
        pivot = i;
    const double magnitude = QMC_PF_COMPLEX_ABS(a[pivot * n + k]);
    if (!device_math::isFiniteBinary64(magnitude))
    {
      fail(status, channel);
      return;
    }
    if (magnitude == 0)
      return;
    if constexpr (CollectPivots)
      minimum_pivot = fmin(minimum_pivot, magnitude);
    if (pivot != k)
    {
      for (std::size_t j = 0; j < n; ++j)
      {
        const auto temporary = a[k * n + j];
        a[k * n + j] = a[pivot * n + j];
        a[pivot * n + j] = temporary;
      }
      phase = make_cuDoubleComplex(-cuCreal(phase), -cuCimag(phase));
      if (perm)
      {
        const auto temporary = perm[k];
        perm[k] = perm[pivot];
        perm[pivot] = temporary;
      }
    }
    phase = QMC_PF_COMPLEX_MUL(phase, make_cuDoubleComplex(cuCreal(a[k * n + k]) / magnitude,
                                             cuCimag(a[k * n + k]) / magnitude));
    log_abs += log(magnitude);
    if (!finite(phase) || !device_math::isFiniteBinary64(log_abs))
    {
      fail(status, channel);
      return;
    }
    for (std::size_t i = k + 1; i < n; ++i)
    {
      a[i * n + k] = QMC_PF_COMPLEX_DIV(a[i * n + k], a[k * n + k]);
      if (!finite(a[i * n + k]))
      {
        fail(status, channel);
        return;
      }
      for (std::size_t j = k + 1; j < n; ++j)
      {
        a[i * n + j] = QMC_PF_COMPLEX_SUB(a[i * n + j], QMC_PF_COMPLEX_MUL(a[i * n + k], a[k * n + j]));
        if (!finite(a[i * n + j]))
        {
          fail(status, channel);
          return;
        }
      }
    }
  }
  for (std::size_t i = 0; i < n * n; ++i)
    if (!finite(a[i]))
    {
      fail(status, channel);
      return;
    }
  const double magnitude = QMC_PF_COMPLEX_ABS(phase);
  if (!device_math::isFiniteBinary64(log_abs) || !finite(phase) || !(magnitude > 0))
  {
    fail(status, channel);
    return;
  }
  metadata[channel] = {make_cuDoubleComplex(cuCreal(phase) / magnitude, cuCimag(phase) / magnitude), log_abs, 0};
  if constexpr (EmitScale)
    solve[channel * n] = make_cuDoubleComplex(scale, 0);
  if (inverse && !inverseFromLu(a, n, scale, perm, inverse + channel * n * n, solve + channel * n))
    fail(status, channel);
  if constexpr (CollectPivots)
    minimum_scaled_pivots[channel] = minimum_pivot;
}

/** Independent columns retain the canonical forward/backward arithmetic order.
 * The private inverse column first holds forward y, then descending rows replace
 * y with x. Scaling is deferred until that complete backward solve has finished.
 * No other column or consumer observes this scratch before the stream boundary.
 */
__global__ void inverseColumns(std::size_t n, const cuDoubleComplex* matrices,
    const ComplexValueChannel* metadata, std::uint64_t* status, cuDoubleComplex* inverses,
    const std::size_t* permutation, const cuDoubleComplex* solve)
{
  const auto channel = std::size_t(blockIdx.x);
  if (metadata[channel].status)
    return;
  const auto* a = matrices + channel * n * n;
  const auto* perm = permutation + channel * n;
  auto* inverse = inverses + channel * n * n;
  const double scale = cuCreal(solve[channel * n]);
  for (std::size_t column = threadIdx.x; column < n; column += blockDim.x)
  {
    bool regular = true;
    for (std::size_t i = 0; i < n; ++i)
    {
      auto value = make_cuDoubleComplex(perm[i] == column ? 1 : 0, 0);
      for (std::size_t j = 0; j < i; ++j)
        value = QMC_PF_COMPLEX_SUB(value, QMC_PF_COMPLEX_MUL(a[i * n + j], inverse[j * n + column]));
      if (!finite(value))
      {
        regular = false;
        break;
      }
      inverse[i * n + column] = value;
    }
    if (regular)
      for (std::size_t i = n; i-- > 0;)
      {
        auto value = inverse[i * n + column];
        for (std::size_t j = i + 1; j < n; ++j)
          value = QMC_PF_COMPLEX_SUB(value, QMC_PF_COMPLEX_MUL(a[i * n + j], inverse[j * n + column]));
        inverse[i * n + column] = QMC_PF_COMPLEX_DIV(value, a[i * n + i]);
        if (!finite(inverse[i * n + column]))
        {
          regular = false;
          break;
        }
      }
    // Higher x entries must remain unscaled until all backward substitutions end.
    if (regular)
      for (std::size_t i = 0; i < n; ++i)
      {
        const auto value = inverse[i * n + column];
        inverse[i * n + column] = make_cuDoubleComplex(cuCreal(value) / scale, cuCimag(value) / scale);
        if (!finite(inverse[i * n + column]))
        {
          regular = false;
          break;
        }
      }
    if (!regular)
      fail(status, channel);
  }
}

/// One block owns one in-place LU, with scalar phase/log and optional inverse-column scale.
/// Every entry keeps the serial k-order and vendor operations; no sums are reordered.
template<bool CollectPivots, bool EmitScale = false>
__global__ void factorCooperative(std::size_t channels, std::size_t n, cuDoubleComplex* matrices,
    ComplexValueChannel* metadata, std::uint64_t* status, double* minimum_scaled_pivots,
    std::size_t* permutation = nullptr, cuDoubleComplex* solve = nullptr)
{
  const auto channel = std::size_t(blockIdx.x), lane = std::size_t(threadIdx.x);
  if (channel >= channels)
    return;
  __shared__ double magnitudes[256];
  __shared__ std::size_t rows[256];
  __shared__ double scale, log_abs, pivot_magnitude;
  __shared__ double minimum_pivot;
  __shared__ cuDoubleComplex phase;
  __shared__ std::size_t pivot;
  __shared__ unsigned bad;
  auto* a = matrices + channel * n * n;
  if constexpr (EmitScale)
    for (auto i = lane; i < n; i += 256)
      permutation[channel * n + i] = i;
  if (!lane)
  {
    metadata[channel] = {make_cuDoubleComplex(0, 0), nodeLog(), 1};
    if constexpr (CollectPivots)
    {
      minimum_scaled_pivots[channel] = 0;
      minimum_pivot = INFINITY;
    }
    bad = 0;
  }
  __syncthreads();
  double local_scale = 0;
  for (auto i = lane; i < n * n; i += 256)
  {
    if (!finite(a[i])) atomicOr(&bad, 1u);
    local_scale = fmax(local_scale, fmax(fabs(cuCreal(a[i])), fabs(cuCimag(a[i]))));
  }
  magnitudes[lane] = local_scale;
  __syncthreads();
  for (unsigned stride = 128; stride; stride >>= 1)
  {
    if (lane < stride) magnitudes[lane] = fmax(magnitudes[lane], magnitudes[lane + stride]);
    __syncthreads();
  }
  if (!lane)
  {
    scale = magnitudes[0];
    if constexpr (!EmitScale)
      if (bad) fail(status, channel);
    phase = make_cuDoubleComplex(1, 0);
    log_abs = double(n) * log(scale);
  }
  __syncthreads();
  if (bad || scale == 0) return;
  for (auto i = lane; i < n * n; i += 256)
    a[i] = make_cuDoubleComplex(cuCreal(a[i]) / scale, cuCimag(a[i]) / scale);
  __syncthreads();
  for (std::size_t k = 0; k < n; ++k)
  {
    double best = -1;
    std::size_t row = n;
    for (auto i = k + lane; i < n; i += 256)
    {
      const auto magnitude = QMC_PF_COMPLEX_ABS(a[i * n + k]);
      if (magnitude > best) { best = magnitude; row = i; }
    }
    magnitudes[lane] = best;
    rows[lane] = row;
    __syncthreads();
    for (unsigned stride = 128; stride; stride >>= 1)
    {
      if (lane < stride && (magnitudes[lane + stride] > magnitudes[lane] ||
          (magnitudes[lane + stride] == magnitudes[lane] && rows[lane + stride] < rows[lane])))
      { magnitudes[lane] = magnitudes[lane + stride]; rows[lane] = rows[lane + stride]; }
      __syncthreads();
    }
    if (!lane)
    {
      pivot = rows[0];
      pivot_magnitude = magnitudes[0];
      if (!device_math::isFiniteBinary64(pivot_magnitude))
      {
        bad = 1;
        if constexpr (!EmitScale)
          fail(status, channel);
      }
      if constexpr (CollectPivots)
        minimum_pivot = fmin(minimum_pivot, pivot_magnitude);
    }
    __syncthreads();
    if (bad || pivot_magnitude == 0) return;
    if (pivot != k)
      for (auto j = lane; j < n; j += 256)
      {
        const auto temporary = a[k * n + j];
        a[k * n + j] = a[pivot * n + j];
        a[pivot * n + j] = temporary;
      }
    __syncthreads();
    if (!lane)
    {
      if (pivot != k)
      {
        phase = make_cuDoubleComplex(-cuCreal(phase), -cuCimag(phase));
        if constexpr (EmitScale)
        {
          auto* perm = permutation + channel * n;
          const auto temporary = perm[k];
          perm[k] = perm[pivot];
          perm[pivot] = temporary;
        }
      }
      phase = QMC_PF_COMPLEX_MUL(phase, make_cuDoubleComplex(cuCreal(a[k * n + k]) / pivot_magnitude,
          cuCimag(a[k * n + k]) / pivot_magnitude));
      log_abs += log(pivot_magnitude);
      if (!finite(phase) || !device_math::isFiniteBinary64(log_abs))
      {
        bad = 1;
        if constexpr (!EmitScale)
          fail(status, channel);
      }
    }
    __syncthreads();
    if (bad) return;
    for (auto i = k + 1 + lane; i < n; i += 256)
    {
      a[i * n + k] = QMC_PF_COMPLEX_DIV(a[i * n + k], a[k * n + k]);
      if (!finite(a[i * n + k])) atomicOr(&bad, 1u);
    }
    __syncthreads();
    if (bad)
    {
      if constexpr (!EmitScale)
        if (!lane) fail(status, channel);
      return;
    }
    const auto trailing = n - k - 1;
    for (auto index = lane; index < trailing * trailing; index += 256)
    {
      const auto i = k + 1 + index / trailing, j = k + 1 + index % trailing;
      a[i * n + j] = QMC_PF_COMPLEX_SUB(a[i * n + j], QMC_PF_COMPLEX_MUL(a[i * n + k], a[k * n + j]));
      if (!finite(a[i * n + j])) atomicOr(&bad, 1u);
    }
    __syncthreads();
    if (bad)
    {
      if constexpr (!EmitScale)
        if (!lane) fail(status, channel);
      return;
    }
  }
  for (auto i = lane; i < n * n; i += 256)
    if (!finite(a[i])) atomicOr(&bad, 1u);
  __syncthreads();
  if (!lane)
  {
    const double magnitude = QMC_PF_COMPLEX_ABS(phase);
    if (bad || !device_math::isFiniteBinary64(log_abs) || !finite(phase) || !(magnitude > 0))
    {
      if constexpr (!EmitScale)
        fail(status, channel);
    }
    else
    {
      metadata[channel] = {make_cuDoubleComplex(cuCreal(phase) / magnitude, cuCimag(phase) / magnitude), log_abs, 0};
      if constexpr (EmitScale)
        solve[channel * n] = make_cuDoubleComplex(scale, 0);
      if constexpr (CollectPivots)
        minimum_scaled_pivots[channel] = minimum_pivot;
    }
  }
}

/// Raw target replay uses only complex loads/stores, retaining signed-zero and nonfinite payloads.
template<bool FailedOnly>
__global__ void copyTargetRaw(std::size_t n, const ComplexValueChannel* metadata,
    const cuDoubleComplex* source, cuDoubleComplex* destination, std::uint8_t* decisions)
{
  const auto channel = std::size_t(blockIdx.x);
  if constexpr (FailedOnly)
    if (metadata[channel].status == 0)
      return;
  if constexpr (!FailedOnly)
    if (!threadIdx.x && decisions)
      decisions[channel] = 0;
  const auto begin = channel * n * n;
  for (auto local = std::size_t(threadIdx.x); local < n * n; local += blockDim.x)
    destination[begin + local] = source[begin + local];
}

/// Restore failed speculative LU from retained providers, never from the inverse owner.
__global__ void restoreFailedPacked(std::size_t n, const ComplexValueChannel* metadata,
    const double* envelope, const double* real, const double* imag, cuDoubleComplex* matrix)
{
  const auto channel = std::size_t(blockIdx.x);
  if (metadata[channel].status == 0)
    return;
  const auto begin = channel * n * n;
  for (auto local = std::size_t(threadIdx.x); local < n * n; local += blockDim.x)
  {
    const auto index = begin + local;
    // Identical to the original pack; canonical scalar replay alone publishes failure.
    matrix[index] = make_cuDoubleComplex(envelope[index] * real[index], envelope[index] * imag[index]);
  }
}

/// Cold opt-in; changing a process environment after its first VALUE call is unsupported.
bool cooperativeFactorEnabled()
{
  static const bool enabled = [] {
    const char* value = std::getenv("PSIFORMER_GPU_COMPLEX_VALUE_COOPERATIVE_LU");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::invalid_argument("PSIFORMER_GPU_COMPLEX_VALUE_COOPERATIVE_LU must be 0 or 1");
  }();
  return enabled;
}

/// Integer-only earliest-event reduction preserves the scalar i,j scan order.
__global__ void pairCheck(std::size_t electrons, std::size_t spin_up, periodic_geometry::Cell cell,
    const double* positions, std::uint64_t* events)
{
  __shared__ unsigned long long earliest;
  if (!threadIdx.x)
    earliest = UINT64_MAX;
  __syncthreads();
  const auto* r = positions + 3 * std::size_t(blockIdx.x) * electrons;
  std::uint64_t local = UINT64_MAX;
  for (std::size_t p = threadIdx.x; p < electrons * electrons; p += blockDim.x)
  {
    const auto i = p / electrons, j = p % electrons;
    if (i >= j || (i < spin_up) != (j < spin_up))
      continue;
    periodic_geometry::Pair pair;
    if (!periodic_geometry::evaluatePair(cell, r + 3 * i, r + 3 * j, pair))
      local = local < 2 * p + 1 ? local : 2 * p + 1;
    else if (pair.radius == 0)
      local = local < 2 * p ? local : 2 * p;
  }
  // Each lane contributes only its earliest integer event, never a numerical status.
  atomicMin(&earliest, static_cast<unsigned long long>(local));
  __syncthreads();
  if (!threadIdx.x)
    events[blockIdx.x] = earliest;
}

/// Mix channels at a common shift; node outputs never require logarithmic derivatives.
__global__ void reduce(std::size_t batch, std::size_t determinants, const ComplexValueChannel* channels,
    std::size_t electrons, std::size_t spin_up, periodic_geometry::Cell cell, const double* positions,
    const double* cusp, ComplexValueResult* result, std::uint64_t* status, std::size_t cusp_stride,
    const std::uint64_t* pair_events)
{
  const auto configuration = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (configuration >= batch)
    return;
  result[configuration] = {make_cuDoubleComplex(0, 0), nodeLog(), 0, 0, 1};
  const auto* records = channels + configuration * determinants;
  double shift = nodeLog();
  std::uint64_t singular = 0;
  for (std::size_t d = 0; d < determinants; ++d)
    if (records[d].status == 1)
      ++singular;
    else
      shift = fmax(shift, records[d].log_abs);
  result[configuration].singular_channels = singular;
  if (singular == determinants)
    return;
  device_determinant::CompensatedSumBinary64 re, im, absolute;
  for (std::size_t d = 0; d < determinants; ++d)
    if (records[d].status == 0)
    {
      const double scale = exp(records[d].log_abs - shift);
      re.add(scale * cuCreal(records[d].phase));
      im.add(scale * cuCimag(records[d].phase));
      absolute.add(scale);
    }
  const double magnitude = hypot(re.value(), im.value());
  if (!device_math::isFiniteBinary64(magnitude) || !device_math::isFiniteBinary64(absolute.value()))
  {
    fail(status, configuration);
    return;
  }
  if (magnitude == 0)
    return;
  // Consume speculative pair records only at the original observable checkpoint.
  if (pair_events)
  {
    const auto event = pair_events[configuration];
    if (event != UINT64_MAX)
    {
      if (event & 1)
        fail(status, configuration);
      return;
    }
  }
  else
  {
    // Preserve the CPU's exact same-spin physical node, including periodic images.
    for (std::size_t i = 0; i < electrons; ++i)
      for (std::size_t j = i + 1; j < electrons; ++j)
        if ((i < spin_up) == (j < spin_up))
        {
          periodic_geometry::Pair pair;
          if (!periodic_geometry::evaluatePair(cell, positions + 3 * (configuration * electrons + i),
                positions + 3 * (configuration * electrons + j), pair))
          {
            fail(status, configuration);
            return;
          }
          if (pair.radius == 0)
            return;
        }
  }
  const double log_abs = shift + log(magnitude) + cusp[configuration * cusp_stride];
  if (!device_math::isFiniteBinary64(log_abs))
  {
    fail(status, configuration);
    return;
  }
  result[configuration] = {make_cuDoubleComplex(re.value() / magnitude, im.value() / magnitude),
      log_abs, magnitude / absolute.value(), singular, 0};
}

/// Reject invalid grid extents before launch independently of checked owner preparation.
unsigned blocks(std::size_t count)
{
  if (!count || (count - 1) / 128 >= std::size_t(std::numeric_limits<int>::max()))
    throw std::length_error("Complex VALUE launch extent is invalid");
  return static_cast<unsigned>((count - 1) / 128 + 1);
}

/// Return the current vendor launch status without hiding synchronization.
Error launched()
{
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}
} // namespace

/// Launch the exact shared periodic real envelope value primitive.
Error launchComplexValueEnvelopes(Stream stream, OpenOrbitalJetLayout layout, periodic_geometry::Cell cell,
    const double* positions, const double* nuclei, const double* pi_up, const double* pi_down,
    const double* zeta_up, const double* zeta_down, double* envelope, std::uint64_t* status)
{
  if ((layout.features.mode != SpatialJetMode::VALUE_ONLY && layout.features.mode != SpatialJetMode::FULL_VGL) ||
      !positions || !nuclei || !pi_up || !pi_down ||
      !zeta_up || !zeta_down || !envelope || !status)
    throw std::invalid_argument("Complex VALUE envelope input is invalid");
  envelopes<<<blocks(layout.features.configuration_count * layout.orbitals.element_count), 128, 0, stream>>>(
      layout, cell, positions, nuclei, pi_up, pi_down, zeta_up, zeta_down, envelope, status);
  return launched();
}

/// Anchor work remains private and stream ordered behind the first tile's existing coordinate upload.
Error launchComplexValueEnvelopeAnchor(Stream stream, OpenOrbitalJetLayout layout, periodic_geometry::Cell cell,
    const double* positions, const double* nuclei, const double* pi_up, const double* pi_down,
    const double* zeta_up, const double* zeta_down, std::size_t moved_row,
    double base_x, double base_y, double base_z, double* anchor, std::uint64_t* anchor_status)
{
  const auto n = layout.features.electron_count;
  if (layout.features.mode != SpatialJetMode::VALUE_ONLY || n < 64 || moved_row >= n ||
      !layout.features.configuration_count || layout.orbital_row_stride != n ||
      layout.orbital_determinant_stride != n * n ||
      layout.orbitals.element_count != layout.determinant_count * n * n ||
      layout.orbitals.configuration_stride != layout.orbitals.element_count ||
      !positions || !nuclei || !pi_up || !pi_down || !zeta_up || !zeta_down || !anchor || !anchor_status)
    throw std::invalid_argument("Complex envelope anchor requires an ordinary VALUE base");
  layout.features.configuration_count = layout.orbitals.configuration_count = 1;
  envelopeAnchor<<<blocks(layout.determinant_count * n * n), 128, 0, stream>>>(layout, cell,
      positions, nuclei, pi_up, pi_down, zeta_up, zeta_down, moved_row,
      base_x, base_y, base_z, anchor, anchor_status);
  return launched();
}

/// Launch only the packed ordinary VALUE layout admitted by the request owner.
Error launchComplexValueVirtualEnvelopes(Stream stream, OpenOrbitalJetLayout layout, periodic_geometry::Cell cell,
    const double* positions, const double* nuclei, const double* pi_up, const double* pi_down,
    const double* zeta_up, const double* zeta_down, const double* anchor,
    const std::size_t* moved_rows, const std::uint64_t* anchor_status,
    double* envelope, std::uint64_t* status)
{
  const auto n = layout.features.electron_count;
  if (layout.features.mode != SpatialJetMode::VALUE_ONLY || n < 64 || !layout.features.configuration_count ||
      layout.orbital_row_stride != n ||
      layout.orbital_determinant_stride != n * n ||
      layout.orbitals.element_count != layout.determinant_count * n * n ||
      layout.orbitals.configuration_stride != layout.orbitals.element_count ||
      !positions || !nuclei || !pi_up || !pi_down || !zeta_up || !zeta_down ||
      !anchor || !moved_rows || !anchor_status || !envelope || !status)
    throw std::invalid_argument("Complex virtual envelope input requires a packed VALUE request");
  virtualEnvelopes<<<blocks(layout.features.configuration_count * layout.orbitals.element_count), 128, 0, stream>>>(
      layout, cell, positions, nuclei, pi_up, pi_down, zeta_up, zeta_down,
      anchor, moved_rows, anchor_status, envelope, status);
  return launched();
}

/// Pack independent real planes into complex storage, never in place.
Error launchComplexValuePack(Stream stream, std::size_t count, const double* envelope,
    const double* real, const double* imag, cuDoubleComplex* matrix, std::uint64_t* status)
{
  if (!envelope || !real || !imag || !matrix || !status)
    throw std::invalid_argument("Complex VALUE pack storage is null");
  pack<<<blocks(count), 128, 0, stream>>>(count, envelope, real, imag, matrix, status);
  return launched();
}

/// Factor the sole complex matrix buffer after its head epoch is consumed.
Error launchComplexValueFactor(Stream stream, std::size_t channels, std::size_t n,
    cuDoubleComplex* matrix, ComplexValueChannel* metadata, std::uint64_t* status,
    cuDoubleComplex* inverse, std::size_t* permutation, cuDoubleComplex* solve)
{
  if (!n || !matrix || !metadata || !status || bool(inverse) != bool(permutation) || bool(inverse) != bool(solve))
    throw std::invalid_argument("Complex VALUE factor storage/size is invalid");
  const auto serial_blocks = blocks(channels);
  if (!inverse && cooperativeFactorEnabled() && channels <= std::size_t(std::numeric_limits<int>::max()))
    factorCooperative<false><<<static_cast<unsigned>(channels), 256, 0, stream>>>(channels, n, matrix, metadata, status, nullptr);
  else
    factor<false><<<serial_blocks, 128, 0, stream>>>(channels, n, matrix, metadata, status, inverse, permutation, solve, nullptr);
  return launched();
}

/// Exact raw-matrix fallback shares the unchanged scalar factor/inverse body.
Error launchComplexTargetFactorReuse(Stream stream, std::size_t channels, std::size_t n,
    cuDoubleComplex* current, cuDoubleComplex* canonical_raw, bool reuse_previous,
    ComplexValueChannel* metadata, std::uint64_t* status, cuDoubleComplex* inverse,
    std::size_t* permutation, cuDoubleComplex* solve, std::uint8_t* decisions)
{
  if (!channels || !n || !current || !canonical_raw || !metadata || !status || !inverse || !permutation || !solve ||
      current == canonical_raw || canonical_raw == inverse || current == inverse)
    throw std::invalid_argument("Target factor reuse requires distinct current/raw/inverse storage");
  const auto elements = spatial_detail::checkedProduct(channels,
      spatial_detail::checkedProduct(n, n, "Target factor matrix extent overflow"),
      "Target factor matrix extent overflow");
  if (elements - 1 > (std::numeric_limits<std::uint64_t>::max() >> 8))
    throw std::overflow_error("Target factor reuse exceeds status indexing");
  factor<false, false, true><<<blocks(channels), 128, 0, stream>>>(channels, n, current, metadata, status,
      inverse, permutation, solve, nullptr, canonical_raw, reuse_previous, decisions);
  return launched();
}

/// Private cold target factorization with an exact existing raw bank and no added owner.
Error launchComplexTargetCooperativeColdFactor(Stream stream, std::size_t channels, std::size_t n,
    cuDoubleComplex* current, cuDoubleComplex* canonical_raw, ComplexValueChannel* metadata,
    std::uint64_t* status, cuDoubleComplex* inverse, std::size_t* permutation,
    cuDoubleComplex* solve, std::uint8_t* decisions)
{
  if (!channels || !n || n > 300 || !current || !canonical_raw || !metadata || !status ||
      !inverse || !permutation || !solve)
    throw std::invalid_argument("Cold target factors require complete distinct raw/LU/inverse storage and n<=300");
  const auto elements = spatial_detail::checkedProduct(channels,
      spatial_detail::checkedProduct(n, n, "Cold target matrix extent overflow"),
      "Cold target matrix extent overflow");
  const auto rows = spatial_detail::checkedProduct(channels, n, "Cold target row extent overflow");
  if (channels > std::size_t(std::numeric_limits<int>::max()) ||
      elements - 1 > (std::numeric_limits<std::uint64_t>::max() >> 8))
    throw std::overflow_error("Cold target factors exceed launch/status bounds");
  const auto bytes = [](std::size_t count, std::size_t size) {
    return spatial_detail::checkedProduct(count, size, "Cold target byte extent overflow");
  };
  const void* owners[]{current, canonical_raw, inverse, metadata, status, permutation, solve, decisions};
  const std::size_t spans[]{bytes(elements, sizeof(cuDoubleComplex)), bytes(elements, sizeof(cuDoubleComplex)),
      bytes(elements, sizeof(cuDoubleComplex)), bytes(channels, sizeof(ComplexValueChannel)), sizeof(std::uint64_t),
      bytes(rows, sizeof(std::size_t)), bytes(rows, sizeof(cuDoubleComplex)), bytes(channels, sizeof(std::uint8_t))};
  const auto count = decisions ? std::size_t(8) : std::size_t(7);
  for (std::size_t i = 0; i < count; ++i)
  {
    const auto begin = reinterpret_cast<std::uintptr_t>(owners[i]);
    if (spans[i] > std::numeric_limits<std::uintptr_t>::max() - begin)
      throw std::overflow_error("Cold target address span overflow");
    for (std::size_t j = 0; j < i; ++j)
    {
      const auto other = reinterpret_cast<std::uintptr_t>(owners[j]);
      if (begin < other + spans[j] && other < begin + spans[i])
        throw std::invalid_argument("Cold target writable owners overlap");
    }
  }
  copyTargetRaw<false><<<static_cast<unsigned>(channels), 256, 0, stream>>>(
      n, metadata, current, canonical_raw, decisions);
  auto result = launched();
  if (result != success)
    return result;
  factorCooperative<false, true><<<static_cast<unsigned>(channels), 256, 0, stream>>>(
      channels, n, current, metadata, status, nullptr, permutation, solve);
  result = launched();
  if (result != success)
    return result;
  copyTargetRaw<true><<<static_cast<unsigned>(channels), 256, 0, stream>>>(
      n, metadata, canonical_raw, current, nullptr);
  result = launched();
  if (result != success)
    return result;
  factor<false, true, false, true><<<blocks(channels), 128, 0, stream>>>(
      channels, n, current, metadata, status, nullptr, permutation, solve, nullptr);
  result = launched();
  if (result != success)
    return result;
  inverseColumns<<<static_cast<unsigned>(channels), 128, 0, stream>>>(
      n, current, metadata, status, inverse, permutation, solve);
  return launched();
}

/// The scalar phase emits only a private scale; the second launch owns columns.
Error launchComplexSpatialInverseColumns(Stream stream, std::size_t channels, std::size_t n,
    cuDoubleComplex* matrix, ComplexValueChannel* metadata, std::uint64_t* status,
    cuDoubleComplex* inverse, std::size_t* permutation, cuDoubleComplex* solve)
{
  if (!channels || !n || !matrix || !metadata || !status || !inverse || !permutation || !solve)
    throw std::invalid_argument("Complex inverse columns require complete existing FULL storage");
  const auto elements = spatial_detail::checkedProduct(channels,
      spatial_detail::checkedProduct(n, n, "Complex inverse matrix extent overflow"),
      "Complex inverse matrix extent overflow");
  if (channels > std::size_t(std::numeric_limits<int>::max()) ||
      n > std::size_t(std::numeric_limits<int>::max()) ||
      elements - 1 > (std::numeric_limits<std::uint64_t>::max() >> 8))
    throw std::overflow_error("Complex inverse columns exceed launch/status bounds");
  const auto serial_blocks = blocks(channels);
  factor<false, true><<<serial_blocks, 128, 0, stream>>>(
      channels, n, matrix, metadata, status, nullptr, permutation, solve, nullptr);
  const auto result = launched();
  if (result != success)
    return result;
  inverseColumns<<<static_cast<unsigned>(channels), 128, 0, stream>>>(
      n, matrix, metadata, status, inverse, permutation, solve);
  return launched();
}

/// Reuse retained packed providers for exact failure replay without another numeric bank.
Error launchComplexScoreCooperativeInverseColumns(Stream stream, std::size_t channels, std::size_t n,
    const double* envelope, const double* real, const double* imag,
    cuDoubleComplex* matrix, ComplexValueChannel* metadata, std::uint64_t* status,
    cuDoubleComplex* inverse, std::size_t* permutation, cuDoubleComplex* solve)
{
  if (!channels || !n || n > 300 || !envelope || !real || !imag || !matrix || !metadata || !status ||
      !inverse || !permutation || !solve)
    throw std::invalid_argument("Cooperative score LU requires complete retained packed providers/storage");
  const auto elements = spatial_detail::checkedProduct(channels,
      spatial_detail::checkedProduct(n, n, "Cooperative score matrix extent overflow"),
      "Cooperative score matrix extent overflow");
  const auto rows = spatial_detail::checkedProduct(channels, n, "Cooperative score row extent overflow");
  if (channels > std::size_t(std::numeric_limits<int>::max()) ||
      elements - 1 > (std::numeric_limits<std::uint64_t>::max() >> 8))
    throw std::overflow_error("Cooperative score LU exceeds launch/status bounds");
  const auto overlap = [](const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    const auto first = reinterpret_cast<std::uintptr_t>(a), second = reinterpret_cast<std::uintptr_t>(b);
    if (a_bytes > std::numeric_limits<std::uintptr_t>::max() - first ||
        b_bytes > std::numeric_limits<std::uintptr_t>::max() - second)
      throw std::overflow_error("Cooperative score address span overflow");
    return first < second + b_bytes && second < first + a_bytes;
  };
  const void* destinations[]{matrix, inverse, metadata, status, permutation, solve};
  const auto bytes = [](std::size_t count, std::size_t element_bytes) {
    return spatial_detail::checkedProduct(count, element_bytes, "Cooperative score byte extent overflow");
  };
  const std::size_t sizes[]{bytes(elements, sizeof(cuDoubleComplex)), bytes(elements, sizeof(cuDoubleComplex)),
      bytes(channels, sizeof(ComplexValueChannel)), sizeof(std::uint64_t),
      bytes(rows, sizeof(std::size_t)), bytes(rows, sizeof(cuDoubleComplex))};
  for (std::size_t i = 0; i < 6; ++i)
  {
    for (const auto* provider : {envelope, real, imag})
      if (overlap(destinations[i], sizes[i], provider, bytes(elements, sizeof(double))))
        throw std::invalid_argument("Cooperative score packed providers alias writable scratch");
    for (std::size_t j = 0; j < i; ++j)
      if (overlap(destinations[i], sizes[i], destinations[j], sizes[j]))
        throw std::invalid_argument("Cooperative score writable owners overlap");
  }
  factorCooperative<false, true><<<static_cast<unsigned>(channels), 256, 0, stream>>>(
      channels, n, matrix, metadata, status, nullptr, permutation, solve);
  auto result = launched();
  if (result != success)
    return result;
  restoreFailedPacked<<<static_cast<unsigned>(channels), 256, 0, stream>>>(
      n, metadata, envelope, real, imag, matrix);
  result = launched();
  if (result != success)
    return result;
  // Singular and failed sentinels both replay the unchanged canonical initialization/body.
  factor<false, true, false, true><<<blocks(channels), 128, 0, stream>>>(
      channels, n, matrix, metadata, status, nullptr, permutation, solve, nullptr);
  result = launched();
  if (result != success)
    return result;
  inverseColumns<<<static_cast<unsigned>(channels), 128, 0, stream>>>(
      n, matrix, metadata, status, inverse, permutation, solve);
  return launched();
}

/// Collect component-max-scaled pivots without altering factor arithmetic or publication.
Error launchComplexValueFactorWithDiagnostics(Stream stream, std::size_t channels, std::size_t n,
    cuDoubleComplex* matrix, ComplexValueChannel* metadata, std::uint64_t* status,
    double* minimum_scaled_pivots)
{
  if (!channels || !n || !matrix || !metadata || !status || !minimum_scaled_pivots)
    throw std::invalid_argument("Complex VALUE diagnostic factor storage/size is invalid");
  if (cooperativeFactorEnabled() && channels <= std::size_t(std::numeric_limits<int>::max()))
    factorCooperative<true><<<static_cast<unsigned>(channels), 256, 0, stream>>>(
        channels, n, matrix, metadata, status, minimum_scaled_pivots);
  else
    factor<true><<<blocks(channels), 128, 0, stream>>>(
        channels, n, matrix, metadata, status, nullptr, nullptr, nullptr, minimum_scaled_pivots);
  return launched();
}

/// Collect pivots while executing the unchanged inverse-enabled scalar LU path.
Error launchComplexSpatialFactorWithDiagnostics(Stream stream, std::size_t channels, std::size_t n,
    cuDoubleComplex* matrix, ComplexValueChannel* metadata, std::uint64_t* status,
    cuDoubleComplex* inverse, std::size_t* permutation, cuDoubleComplex* solve,
    double* minimum_scaled_pivots)
{
  if (!channels || !n || !matrix || !metadata || !status || !inverse || !permutation || !solve || !minimum_scaled_pivots)
    throw std::invalid_argument("Complex FULL diagnostic factor storage/size is invalid");
  factor<true><<<blocks(channels), 128, 0, stream>>>(
      channels, n, matrix, metadata, status, inverse, permutation, solve, minimum_scaled_pivots);
  return launched();
}

/// Preflight every launch/index bound without publishing numerical failure early.
Error launchComplexValuePairCheck(Stream stream, std::size_t batch, std::size_t electrons,
    std::size_t spin_up, periodic_geometry::Cell cell, const double* positions, std::uint64_t* events)
{
  if (!batch || batch > std::size_t(std::numeric_limits<int>::max()) || !electrons ||
      spin_up > electrons || !positions || !events)
    throw std::invalid_argument("Complex VALUE pair-check input is invalid");
  if (electrons > (std::numeric_limits<std::uint64_t>::max() / 2) / electrons ||
      electrons > std::numeric_limits<std::size_t>::max() / 3 / batch)
    throw std::overflow_error("Complex VALUE pair-check index exceeds admission");
  pairCheck<<<static_cast<unsigned>(batch), 128, 0, stream>>>(electrons, spin_up, cell, positions, events);
  return launched();
}

/// Publish only the core amplitude and numerical channel diagnostics.
Error launchComplexValueReduce(Stream stream, std::size_t batch, std::size_t determinants,
    std::size_t electrons, std::size_t spin_up, periodic_geometry::Cell cell, const double* positions,
    const ComplexValueChannel* metadata, const double* cusp, ComplexValueResult* result, std::uint64_t* status,
    std::size_t cusp_stride, const std::uint64_t* pair_events)
{
  if (!determinants || !electrons || spin_up > electrons || !positions || !metadata || !cusp || !result || !status || !cusp_stride)
    throw std::invalid_argument("Complex VALUE reduction input is invalid");
  reduce<<<blocks(batch), 128, 0, stream>>>(batch, determinants, metadata,
      electrons, spin_up, cell, positions, cusp, result, status, cusp_stride, pair_events);
  return launched();
}
} // namespace qmcplusplus::psiformer::device
#undef QMC_PF_COMPLEX_ABS
#undef QMC_PF_COMPLEX_MUL
#undef QMC_PF_COMPLEX_DIV
#undef QMC_PF_COMPLEX_SUB
