//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include "PsiFormerDeviceComplexValueKernels.h"
#include "QMCWaveFunctions/PsiFormer/PsiFormerOpenOrbital.h"
#include <cmath>
#include <limits>
#include <stdexcept>

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

/// Serial-per-channel scaled partial-pivot LU; zero pivots are singular, not errors.
__global__ void factor(std::size_t channels, std::size_t n, cuDoubleComplex* matrices,
    ComplexValueChannel* metadata, std::uint64_t* status)
{
  const auto channel = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel >= channels)
    return;
  auto* a = matrices + channel * n * n;
  metadata[channel] = {make_cuDoubleComplex(0, 0), nodeLog(), 1};
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
    if (pivot != k)
    {
      for (std::size_t j = 0; j < n; ++j)
      {
        const auto temporary = a[k * n + j];
        a[k * n + j] = a[pivot * n + j];
        a[pivot * n + j] = temporary;
      }
      phase = make_cuDoubleComplex(-cuCreal(phase), -cuCimag(phase));
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
}

/// Mix channels at a common shift; node outputs never require logarithmic derivatives.
__global__ void reduce(std::size_t batch, std::size_t determinants, const ComplexValueChannel* channels,
    std::size_t electrons, std::size_t spin_up, periodic_geometry::Cell cell, const double* positions,
    const double* cusp, ComplexValueResult* result, std::uint64_t* status)
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
  const double log_abs = shift + log(magnitude) + cusp[configuration];
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
  if (layout.features.mode != SpatialJetMode::VALUE_ONLY || !positions || !nuclei || !pi_up || !pi_down ||
      !zeta_up || !zeta_down || !envelope || !status)
    throw std::invalid_argument("Complex VALUE envelope input is invalid");
  envelopes<<<blocks(spatialJetSpanElements(layout.orbitals)), 128, 0, stream>>>(
      layout, cell, positions, nuclei, pi_up, pi_down, zeta_up, zeta_down, envelope, status);
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
    cuDoubleComplex* matrix, ComplexValueChannel* metadata, std::uint64_t* status)
{
  if (!n || !matrix || !metadata || !status)
    throw std::invalid_argument("Complex VALUE factor storage/size is invalid");
  factor<<<blocks(channels), 128, 0, stream>>>(channels, n, matrix, metadata, status);
  return launched();
}

/// Publish only the core amplitude and numerical channel diagnostics.
Error launchComplexValueReduce(Stream stream, std::size_t batch, std::size_t determinants,
    std::size_t electrons, std::size_t spin_up, periodic_geometry::Cell cell, const double* positions,
    const ComplexValueChannel* metadata, const double* cusp, ComplexValueResult* result, std::uint64_t* status)
{
  if (!determinants || !electrons || spin_up > electrons || !positions || !metadata || !cusp || !result || !status)
    throw std::invalid_argument("Complex VALUE reduction input is invalid");
  reduce<<<blocks(batch), 128, 0, stream>>>(batch, determinants, metadata,
      electrons, spin_up, cell, positions, cusp, result, status);
  return launched();
}
} // namespace qmcplusplus::psiformer::device
#undef QMC_PF_COMPLEX_ABS
#undef QMC_PF_COMPLEX_MUL
#undef QMC_PF_COMPLEX_DIV
#undef QMC_PF_COMPLEX_SUB
