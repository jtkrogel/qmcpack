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

/// Preserve scalar normalization and failure prefixes while distributing independent entries.
__global__ void weightedInversesCooperative(std::size_t d, std::size_t n, const ComplexValueChannel* channels,
    cuDoubleComplex* inverse, const double* cusp, ComplexValueResult* amplitude, std::uint64_t* status)
{
  __shared__ double shift, log_total, log_weights[16];
  __shared__ cuDoubleComplex phase, ratios[16];
  __shared__ unsigned long long first_bad;
  __shared__ bool rejected;
  const auto count = d * n * n;
  if (threadIdx.x == 0)
  {
    rejected = false;
    first_bad = ~0ULL;
    shift = -DBL_MAX;
    // Metadata precedes every inverse check, exactly as in the scalar entry.
    for (std::size_t channel = 0; channel < d; ++channel)
    {
      if (channels[channel].status || amplitude->status || !device_math::isFiniteBinary64(channels[channel].log_abs) ||
          !device_math::isFiniteBinary64(cuCreal(channels[channel].phase)) ||
          !device_math::isFiniteBinary64(cuCimag(channels[channel].phase)))
      {
        fail(status, channel);
        rejected = true;
        break;
      }
      shift = fmax(shift, channels[channel].log_abs);
    }
  }
  __syncthreads();
  if (rejected) return;

  // Admit the whole untouched inverse epoch, selecting the canonical first index.
  for (std::size_t i = threadIdx.x; i < count; i += blockDim.x)
    if (!device_math::isFiniteBinary64(cuCreal(inverse[i])) || !device_math::isFiniteBinary64(cuCimag(inverse[i])))
      atomicMin(&first_bad, static_cast<unsigned long long>(i));
  __syncthreads();
  if (first_bad != ~0ULL)
  {
    if (threadIdx.x == 0) fail(status, first_bad);
    return;
  }
  if (threadIdx.x == 0)
  {
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
      rejected = true;
    }
    else
    {
      const auto ur = re.value() / scale, ui = im.value() / scale, norm = hypot(ur, ui);
      phase = make_cuDoubleComplex(ur / norm, ui / norm);
      log_total = log(scale) + log(norm);
      for (std::size_t channel = 0; channel < d; ++channel)
      {
        log_weights[channel] = (channels[channel].log_abs - shift) - log_total;
        ratios[channel] = QMC_PF_DIV(channels[channel].phase, phase);
      }
    }
  }
  __syncthreads();
  if (rejected) return;

  // No writes until every weighted product admits; failed replay needs original inputs.
  for (std::size_t i = threadIdx.x; i < count; i += blockDim.x)
  {
    cuDoubleComplex candidate;
    const auto channel = i / (n * n);
    if (!complex_detail::weightedProduct(log_weights[channel], ratios[channel], inverse[i], candidate))
      atomicMin(&first_bad, static_cast<unsigned long long>(i));
  }
  __syncthreads();
  if (first_bad != ~0ULL)
  {
    // Reproduce the scalar failed prefix INCLUDING its failing output element.
    if (threadIdx.x == 0)
      for (std::size_t channel = 0; channel < d; ++channel)
        for (std::size_t i = channel * n * n; i < (channel + 1) * n * n; ++i)
          if (!complex_detail::weightedProduct(log_weights[channel], ratios[channel], inverse[i], inverse[i]))
          {
            fail(status, i);
            return;
          }
    return;
  }

  // Independent output arithmetic is identical; publication follows the complete store barrier.
  for (std::size_t i = threadIdx.x; i < count; i += blockDim.x)
  {
    const auto channel = i / (n * n);
    complex_detail::weightedProduct(log_weights[channel], ratios[channel], inverse[i], inverse[i]);
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    amplitude->phase = phase;
    amplitude->log_abs = shift + log_total + cusp[0];
    if (!device_math::isFiniteBinary64(amplitude->log_abs)) fail(status, 0);
  }
}

/// Metadata has canonical priority over every inverse-entry failure.
__global__ void weightedGridBegin(std::size_t d, const ComplexValueChannel* channels,
    const ComplexValueResult* amplitude, std::uint64_t* status, ComplexScoreWeightedGridState* state)
{
  state->first_bad = ~std::uint64_t(0);
  state->rejected = 0;
  state->shift = -DBL_MAX;
  for (std::size_t channel = 0; channel < d; ++channel)
  {
    if (channels[channel].status || amplitude->status || !device_math::isFiniteBinary64(channels[channel].log_abs) ||
        !device_math::isFiniteBinary64(cuCreal(channels[channel].phase)) ||
        !device_math::isFiniteBinary64(cuCimag(channels[channel].phase)))
    {
      fail(status, channel);
      state->rejected = 1;
      return;
    }
    state->shift = fmax(state->shift, channels[channel].log_abs);
  }
}

/// Globally admit the untouched inverse before scalar determinant normalization.
__global__ void weightedGridFinite(std::size_t count, const cuDoubleComplex* inverse,
    ComplexScoreWeightedGridState* state)
{
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (state->rejected || i >= count) return;
  if (!device_math::isFiniteBinary64(cuCreal(inverse[i])) || !device_math::isFiniteBinary64(cuCimag(inverse[i])))
    atomicMin(reinterpret_cast<unsigned long long*>(&state->first_bad), static_cast<unsigned long long>(i));
}

/// Keep the original increasing-channel compensated normalization and per-channel arithmetic.
__global__ void weightedGridNormalize(std::size_t d, const ComplexValueChannel* channels,
    std::uint64_t* status, ComplexScoreWeightedGridState* state)
{
  if (state->rejected) return;
  if (state->first_bad != ~std::uint64_t(0))
  {
    fail(status, state->first_bad);
    state->rejected = 1;
    return;
  }
  device_determinant::CompensatedSumBinary64 re, im;
  for (std::size_t channel = 0; channel < d; ++channel)
  {
    const auto factor = exp(channels[channel].log_abs - state->shift);
    re.add(factor * cuCreal(channels[channel].phase));
    im.add(factor * cuCimag(channels[channel].phase));
  }
  const auto scale = fmax(fabs(re.value()), fabs(im.value()));
  if (!(scale > 0) || !device_math::isFiniteBinary64(re.value()) || !device_math::isFiniteBinary64(im.value()))
  {
    fail(status, 0);
    state->rejected = 1;
    return;
  }
  const auto ur = re.value() / scale, ui = im.value() / scale, norm = hypot(ur, ui);
  state->phase = make_cuDoubleComplex(ur / norm, ui / norm);
  state->log_total = log(scale) + log(norm);
  for (std::size_t channel = 0; channel < d; ++channel)
  {
    state->log_weights[channel] = (channels[channel].log_abs - state->shift) - state->log_total;
    state->ratios[channel] = QMC_PF_DIV(channels[channel].phase, state->phase);
  }
}

/// No mutation until ALL independent weighted products have admitted.
__global__ void weightedGridValidate(std::size_t d, std::size_t n, const cuDoubleComplex* inverse,
    ComplexScoreWeightedGridState* state)
{
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (state->rejected || i >= d * n * n) return;
  const auto channel = i / (n * n);
  cuDoubleComplex candidate;
  if (!complex_detail::weightedProduct(state->log_weights[channel], state->ratios[channel], inverse[i], candidate))
    atomicMin(reinterpret_cast<unsigned long long*>(&state->first_bad), static_cast<unsigned long long>(i));
}

/// Successful entries are disjoint; failure replays the original prefix on untouched inputs.
__global__ void weightedGridStore(std::size_t d, std::size_t n, cuDoubleComplex* inverse,
    std::uint64_t* status, const ComplexScoreWeightedGridState* state)
{
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (state->rejected) return;
  if (state->first_bad != ~std::uint64_t(0))
  {
    if (index == 0)
      for (std::size_t channel = 0; channel < d; ++channel)
        for (std::size_t i = channel * n * n; i < (channel + 1) * n * n; ++i)
          if (!complex_detail::weightedProduct(state->log_weights[channel], state->ratios[channel], inverse[i], inverse[i]))
          {
            fail(status, i);
            return;
          }
    return;
  }
  if (index < d * n * n)
  {
    const auto channel = index / (n * n);
    complex_detail::weightedProduct(state->log_weights[channel], state->ratios[channel], inverse[index], inverse[index]);
  }
}

/// A separate launch is the required GLOBAL store barrier before amplitude publication.
__global__ void weightedGridPublish(const double* cusp, ComplexValueResult* amplitude,
    std::uint64_t* status, const ComplexScoreWeightedGridState* state)
{
  if (state->rejected || state->first_bad != ~std::uint64_t(0)) return;
  amplitude->phase = state->phase;
  amplitude->log_abs = state->shift + state->log_total + cusp[0];
  if (!device_math::isFiniteBinary64(amplitude->log_abs)) fail(status, 0);
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

/// Raw analytic seeds preserve all orbital columns and select only the spin row.
__global__ void orbitalHeadSeed(std::size_t d, std::size_t n, std::size_t spin_up,
    const cuDoubleComplex* cotangents, const double* envelope, const double* real, const double* imag,
    std::size_t component, std::size_t spin, double* seed, std::uint64_t* status)
{
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= d * n * n)
    return;
  const auto row = (i / n) % n, column = i % n, channel = i / (n * n);
  const auto c = cotangents[i];
  const auto target = component == 2 ? i : (row * d + channel) * n + column;
  const auto value = component == 2 ? cuCreal(QMC_PF_MUL(c, make_cuDoubleComplex(real[i], imag[i]))) :
      envelope[i] * (component == 0 ? cuCreal(c) : -cuCimag(c));
  seed[target] = component != 2 && std::size_t(row >= spin_up) != spin ? 0 : value;
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

/// One existing launch and fixed shared scratch, with the general scalar route preserved.
Error launchComplexScoreWeightedInversesCooperative(Stream stream, std::size_t d, std::size_t n,
    const ComplexValueChannel* channels, cuDoubleComplex* inverse, const double* cusp,
    ComplexValueResult* amplitude, std::uint64_t* status)
{
  if (n < 64 || n > 300 || d == 0 || d > 16)
    return launchComplexScoreWeightedInverses(stream, d, n, channels, inverse, cusp, amplitude, status);
  weightedInversesCooperative<<<1, 128, 0, stream>>>(d, n, channels, inverse, cusp, amplitude, status);
  return launched();
}

/// Preserve the old entries while distributing only independent inverse output work.
Error launchComplexScoreWeightedInversesGrid(Stream stream, std::size_t d, std::size_t n,
    const ComplexValueChannel* channels, cuDoubleComplex* inverse, const double* cusp,
    ComplexValueResult* amplitude, std::uint64_t* status,
    ComplexScoreWeightedGridState* state, std::size_t state_count)
{
  if (!n || !d)
    throw std::invalid_argument("Weighted grid dimensions must be positive");
  if (n < 100 || n > 300 || d > 16)
    return launchComplexScoreWeightedInversesCooperative(stream, d, n, channels, inverse, cusp, amplitude, status);
  if (!channels || !inverse || !cusp || !amplitude || !status || !state || state_count < 1)
    throw std::invalid_argument("Weighted grid requires complete inputs and one private state");
  const auto count = d * n * n;
  const auto overlap = [](const void* a, std::size_t bytes, const void* b, std::size_t other) {
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return x <= y ? y - x < bytes : x - y < other;
  };
  if (overlap(state, sizeof(*state), inverse, count * sizeof(*inverse)) ||
      overlap(state, sizeof(*state), channels, d * sizeof(*channels)) ||
      overlap(state, sizeof(*state), cusp, sizeof(*cusp)) ||
      overlap(state, sizeof(*state), amplitude, sizeof(*amplitude)) ||
      overlap(state, sizeof(*state), status, sizeof(*status)))
    throw std::invalid_argument("Weighted grid private state aliases an input or public output");
  const auto blocks = static_cast<unsigned>((count + 127) / 128);
  weightedGridBegin<<<1, 1, 0, stream>>>(d, channels, amplitude, status, state);
  auto error = launched(); if (error != 0) return error;
  weightedGridFinite<<<blocks, 128, 0, stream>>>(count, inverse, state);
  error = launched(); if (error != 0) return error;
  weightedGridNormalize<<<1, 1, 0, stream>>>(d, channels, status, state);
  error = launched(); if (error != 0) return error;
  weightedGridValidate<<<blocks, 128, 0, stream>>>(d, n, inverse, state);
  error = launched(); if (error != 0) return error;
  weightedGridStore<<<blocks, 128, 0, stream>>>(d, n, inverse, status, state);
  error = launched(); if (error != 0) return error;
  weightedGridPublish<<<1, 1, 0, stream>>>(cusp, amplitude, status, state);
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

/// Launch raw head/envelope seeding without any determinant or cusp product.
Error launchComplexOrbitalHeadSeed(Stream stream, std::size_t d, std::size_t n, std::size_t spin_up,
    const cuDoubleComplex* cotangents, const double* envelope, const double* real, const double* imag,
    std::size_t component, std::size_t spin, double* seed, std::uint64_t* status)
{
  orbitalHeadSeed<<<static_cast<unsigned>((d * n * n + 127) / 128), 128, 0, stream>>>(
      d, n, spin_up, cotangents, envelope, real, imag, component, spin, seed, status);
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
