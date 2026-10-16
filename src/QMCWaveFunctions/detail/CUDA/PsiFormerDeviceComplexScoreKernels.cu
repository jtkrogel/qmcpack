//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include "PsiFormerDeviceComplexScoreKernels.h"
#include "PsiFormerDeviceComplexWeightedProducts.h"
#include <cfloat>
#include <stdexcept>
#ifdef QMC_CUDA2HIP
#define QMC_PF_MUL hipCmul
#define QMC_PF_DIV hipCdiv
#else
#define QMC_PF_MUL cuCmul
#define QMC_PF_DIV cuCdiv
#endif

namespace qmcplusplus::psiformer::device
{
namespace
{
/// Record numerical failure using the owner's preflight-checked index encoding.
__device__ void fail(std::uint64_t* status, std::size_t index)
{ atomicMin(reinterpret_cast<unsigned long long*>(status), static_cast<unsigned long long>((std::uint64_t(index) << 8) | 1)); }

/// Admit ALL inverses, normalize a possibly subnormal denominator, then consume I.
__global__ void weightedInverses(std::size_t d, std::size_t n, const ComplexValueChannel* channels,
    cuDoubleComplex* inverse, const double* cusp, ComplexValueResult* amplitude, std::uint64_t* status)
{
  double shift = -DBL_MAX;
  for (std::size_t channel = 0; channel < d; ++channel)
  {
    if (channels[channel].status || amplitude->status || !device_math::isFiniteBinary64(channels[channel].log_abs) ||
        !device_math::isFiniteBinary64(cuCreal(channels[channel].phase)) || !device_math::isFiniteBinary64(cuCimag(channels[channel].phase)))
    {
      fail(status, channel);
      return;
    }
    shift = fmax(shift, channels[channel].log_abs);
  }
  for (std::size_t i = 0; i < d * n * n; ++i)
    if (!device_math::isFiniteBinary64(cuCreal(inverse[i])) || !device_math::isFiniteBinary64(cuCimag(inverse[i])))
    {
      fail(status, i);
      return;
    }
  device_determinant::CompensatedSumBinary64 re, im;
  for (std::size_t channel = 0; channel < d; ++channel)
  {
    const auto factor = exp(channels[channel].log_abs - shift);
    re.add(factor * cuCreal(channels[channel].phase));
    im.add(factor * cuCimag(channels[channel].phase));
  }
  const auto scale = fmax(fabs(re.value()), fabs(im.value()));
  if (!(scale > 0) || !device_math::isFiniteBinary64(re.value()) || !device_math::isFiniteBinary64(im.value()))
  {
    fail(status, 0);
    return;
  }
  const auto ur = re.value() / scale, ui = im.value() / scale, norm = hypot(ur, ui);
  const auto phase = make_cuDoubleComplex(ur / norm, ui / norm);
  const auto log_total = log(scale) + log(norm);
  // No narrowed weight: combine its logarithm with EACH inverse component first.
  for (std::size_t channel = 0; channel < d; ++channel)
  {
    const auto log_weight = (channels[channel].log_abs - shift) - log_total;
    const auto ratio = QMC_PF_DIV(channels[channel].phase, phase);
    for (std::size_t i = channel * n * n; i < (channel + 1) * n * n; ++i)
      if (!complex_detail::weightedProduct(log_weight, ratio, inverse[i], inverse[i]))
      {
        fail(status, i);
        return;
      }
  }
  amplitude->phase = phase;
  amplitude->log_abs = shift + log_total + cusp[0];
  if (!device_math::isFiniteBinary64(amplitude->log_abs))
    fail(status, 0);
}

/// Reuse one realM scratch for four spin/head affine seeds, then the envelope seed.
__global__ void headSeed(std::size_t d, std::size_t n, std::size_t spin_up,
    const cuDoubleComplex* inverse, const double* envelope, const double* real, const double* imag,
    std::size_t objective, std::size_t component, std::size_t spin, double* seed, std::uint64_t* status)
{
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= d * n * n)
    return;
  const auto row = (i / n) % n, column = i % n, channel = i / (n * n);
  const auto h = inverse[channel * n * n + column * n + row];
  auto value = component == 2 ? QMC_PF_MUL(h, make_cuDoubleComplex(real[i], imag[i])) :
      make_cuDoubleComplex(envelope[i] * cuCreal(h), envelope[i] * cuCimag(h));
  if (component == 1)
    value = make_cuDoubleComplex(-cuCimag(value), cuCreal(value));
  const auto target = component == 2 ? i : (row * d + channel) * n + column;
  seed[target] = component != 2 && std::size_t(row >= spin_up) != spin ? 0 : (objective ? cuCimag(value) : cuCreal(value));
  if (!device_math::isFiniteBinary64(seed[target]))
    fail(status, i);
}

/// First pass clears imaginaryP; second pass completes it, both while unpublished.
__global__ void pack(std::size_t p, std::size_t component, const double* response,
    cuDoubleComplex* score, std::uint64_t* status)
{
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= p)
    return;
  if (!device_math::isFiniteBinary64(response[i]))
    fail(status, i);
  score[i] = component ? make_cuDoubleComplex(cuCreal(score[i]), response[i]) : make_cuDoubleComplex(response[i], 0);
}

/// Return the vendor launch status, with synchronization left to the owning transaction.
Error launched()
{
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}
} // namespace

/// The caller has admitted dimensions, buffers and the complete inverse epoch.
Error launchComplexScoreWeightedInverses(Stream stream, std::size_t d, std::size_t n,
    const ComplexValueChannel* channels, cuDoubleComplex* inverse, const double* cusp,
    ComplexValueResult* amplitude, std::uint64_t* status)
{
  weightedInverses<<<1, 1, 0, stream>>>(d, n, channels, inverse, cusp, amplitude, status);
  return launched();
}

/// Consume one selected cotangent into the shared real scratch without allocating.
Error launchComplexScoreHeadSeed(Stream stream, std::size_t d, std::size_t n, std::size_t spin_up,
    const cuDoubleComplex* inverse, const double* envelope, const double* real, const double* imag,
    std::size_t objective, std::size_t component, std::size_t spin, double* seed, std::uint64_t* status)
{
  headSeed<<<static_cast<unsigned>((d * n * n + 127) / 128), 128, 0, stream>>>(
      d, n, spin_up, inverse, envelope, real, imag, objective, component, spin, seed, status);
  return launched();
}

/// Copy a completed finite realP objective into the complex publication buffer.
Error launchComplexScorePack(Stream stream, std::size_t p, std::size_t component,
    const double* response, cuDoubleComplex* score, std::uint64_t* status)
{
  pack<<<static_cast<unsigned>((p + 127) / 128), 128, 0, stream>>>(p, component, response, score, status);
  return launched();
}
} // namespace qmcplusplus::psiformer::device
#undef QMC_PF_MUL
#undef QMC_PF_DIV
