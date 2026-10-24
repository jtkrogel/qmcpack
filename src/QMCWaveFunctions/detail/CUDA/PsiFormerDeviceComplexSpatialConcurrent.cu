//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////

/** @file PsiFormerDeviceComplexSpatialConcurrent.cu
 * @brief Concurrent independent products with canonical scalar trace combinations.
 * Private slab scratch never grants model or result-publication authority.
 */
#include "PsiFormerDeviceComplexSpatialConcurrent.h"
#include "PsiFormerDeviceComplexWeightedProducts.h"
#include <cfloat>

#ifdef QMC_CUDA2HIP
#define QMC_PF_TRACE_MUL hipCmul
#define QMC_PF_TRACE_DIV hipCdiv
#define QMC_PF_TRACE_SUB hipCsub
#else
#define QMC_PF_TRACE_MUL cuCmul
#define QMC_PF_TRACE_DIV cuCdiv
#define QMC_PF_TRACE_SUB cuCsub
#endif

namespace qmcplusplus::psiformer::device
{
namespace
{
/// Preserve componentwise compensated ordinary complex summation order.
struct ComplexSum
{
  device_determinant::CompensatedSumBinary64 re, im;

  /// Add one ordinary complex term without mixing its Cartesian reductions.
  __device__ void add(cuDoubleComplex z)
  {
    re.add(cuCreal(z));
    im.add(cuCimag(z));
  }

  /// Return the independently compensated Cartesian totals.
  __device__ cuDoubleComplex value() const
  {
    return make_cuDoubleComplex(re.value(), im.value());
  }
};

/// Admit intermediate Cartesian components with the existing binary64 predicate.
__device__ bool finite(cuDoubleComplex z)
{
  return device_math::isFiniteBinary64(cuCreal(z)) && device_math::isFiniteBinary64(cuCimag(z));
}

/// Record a configuration failure without publishing any private partial output.
__device__ void fail(std::uint64_t* status, std::size_t b)
{
  atomicMin(reinterpret_cast<unsigned long long*>(status), (static_cast<unsigned long long>(b) << 8) | 1);
}

/// Keep the exact scalar i,j order and transposed pairing, without conjugation.
__device__ cuDoubleComplex trace(const cuDoubleComplex* inverse, const cuDoubleComplex* derivative, std::size_t n)
{
  ComplexSum sum;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      sum.add(QMC_PF_TRACE_MUL(inverse[i * n + j], derivative[j * n + i]));
  return sum.value();
}

/// Each independent determinant/target owns one N*N product and four channel terms.
__global__ void channelKernel(std::size_t batch, std::size_t determinants, std::size_t n,
    std::size_t targets, std::size_t begin, std::size_t live, std::size_t capacity,
    const cuDoubleComplex* inverse, const cuDoubleComplex* head, cuDoubleComplex* product,
    cuDoubleComplex* terms, std::uint64_t* status)
{
  const auto cell = std::size_t(blockIdx.x), d = cell % determinants;
  const auto local = (cell / determinants) % live, b = cell / (determinants * live);
  const auto e = begin + local, nn = n * n, matrix_span = determinants * nn, m = batch * matrix_span;
  const auto offset = b * matrix_span + d * nn, slot = (b * capacity + local) * determinants + d;
  const auto* matrix_inverse = inverse + offset;
  auto* scratch = product + slot * nn;
  auto* channel_terms = terms + 4 * slot;
  ComplexSum channel;
  if (threadIdx.x == 0)
    channel.add(trace(matrix_inverse, head + (1 + 3 * targets + e) * m + offset, n));
  for (std::size_t mu = 0; mu < 3; ++mu)
  {
    const auto* derivative = head + (1 + 3 * e + mu) * m + offset;
    for (std::size_t entry = threadIdx.x; entry < nn; entry += blockDim.x)
    {
      const auto row = entry / n, column = entry % n;
      ComplexSum sum;
      for (std::size_t k = 0; k < n; ++k)
        sum.add(QMC_PF_TRACE_MUL(matrix_inverse[row * n + k], derivative[k * n + column]));
      scratch[entry] = sum.value();
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
      const auto t = trace(matrix_inverse, derivative, n);
      channel_terms[mu] = t;
      channel.add(QMC_PF_TRACE_SUB(QMC_PF_TRACE_MUL(t, t), trace(scratch, scratch, n)));
      if (!finite(t))
        fail(status, b);
    }
    // All signed-square reads finish before this block reuses its product.
    __syncthreads();
  }
  if (threadIdx.x == 0)
  {
    channel_terms[3] = channel.value();
    if (!finite(channel.value()))
      fail(status, b);
  }
}

/// Preserve the existing determinant normalization and scalar compensated D order.
__device__ bool combination(std::size_t determinants, const ComplexValueChannel* records,
    const ComplexValueResult& amplitude, double& shift, double& log_total, cuDoubleComplex& phase)
{
  shift = -DBL_MAX;
  for (std::size_t d = 0; d < determinants; ++d)
  {
    if (records[d].status != 0 || amplitude.status != 0 || !finite(records[d].phase) ||
        !device_math::isFiniteBinary64(records[d].log_abs) || !finite(amplitude.phase) ||
        !device_math::isFiniteBinary64(amplitude.log_abs) ||
        fmax(fabs(cuCreal(amplitude.phase)), fabs(cuCimag(amplitude.phase))) == 0)
      return false;
    shift = fmax(shift, records[d].log_abs);
  }
  ComplexSum total;
  for (std::size_t d = 0; d < determinants; ++d)
  {
    const auto scale = exp(records[d].log_abs - shift);
    total.add(make_cuDoubleComplex(scale * cuCreal(records[d].phase), scale * cuCimag(records[d].phase)));
  }
  const auto denominator = total.value();
  const auto scale = fmax(fabs(cuCreal(denominator)), fabs(cuCimag(denominator)));
  if (!finite(denominator) || !(scale > 0))
    return false;
  const auto re = cuCreal(denominator) / scale, im = cuCimag(denominator) / scale;
  const auto norm = hypot(re, im);
  log_total = log(scale) + log(norm);
  phase = make_cuDoubleComplex(re / norm, im / norm);
  return device_math::isFiniteBinary64(log_total) && finite(phase);
}

/// Independent physical targets combine channels canonically, then add the existing cusp.
__global__ void combineKernel(std::size_t determinants, std::size_t targets, std::size_t output_electrons,
    std::size_t begin, std::size_t live, std::size_t capacity, const ComplexValueChannel* channels,
    const ComplexValueResult* amplitude, const cuDoubleComplex* terms, const double* cusp,
    cuDoubleComplex* gradient, cuDoubleComplex* lap_log, cuDoubleComplex* lap_ratio, std::uint64_t* status)
{
  if (threadIdx.x != 0)
    return;
  const auto b = std::size_t(blockIdx.x) / live, local = std::size_t(blockIdx.x) % live;
  const auto e = begin + local, planes = 1 + 4 * targets;
  const auto* records = channels + b * determinants;
  const auto* values = terms + 4 * (b * capacity + local) * determinants;
  double shift, log_total;
  cuDoubleComplex phase;
  if (!combination(determinants, records, amplitude[b], shift, log_total, phase))
  {
    fail(status, b);
    return;
  }
  cuDoubleComplex core_gradient[3];
  for (std::size_t mu = 0; mu < 3; ++mu)
  {
    ComplexSum sum;
    for (std::size_t d = 0; d < determinants; ++d)
    {
      const auto term = values[4 * d + mu];
      cuDoubleComplex weighted;
      if (!finite(term) || !complex_detail::weightedProduct((records[d].log_abs - shift) - log_total,
          QMC_PF_TRACE_DIV(records[d].phase, phase), term, weighted))
      {
        fail(status, b);
        return;
      }
      sum.add(weighted);
    }
    core_gradient[mu] = sum.value();
  }
  ComplexSum ratio;
  for (std::size_t d = 0; d < determinants; ++d)
  {
    const auto term = values[4 * d + 3];
    cuDoubleComplex weighted;
    if (!finite(term) || !complex_detail::weightedProduct((records[d].log_abs - shift) - log_total,
        QMC_PF_TRACE_DIV(records[d].phase, phase), term, weighted))
    {
      fail(status, b);
      return;
    }
    ratio.add(weighted);
  }
  ComplexSum core_square, physical_square;
  for (std::size_t mu = 0; mu < 3; ++mu)
  {
    auto g = core_gradient[mu];
    core_square.add(QMC_PF_TRACE_MUL(g, g));
    g = make_cuDoubleComplex(cuCreal(g) + cusp[b * planes + 1 + 3 * e + mu], cuCimag(g));
    physical_square.add(QMC_PF_TRACE_MUL(g, g));
    gradient[3 * (b * output_electrons + e) + mu] = g;
    if (!finite(g))
      fail(status, b);
  }
  auto l = QMC_PF_TRACE_SUB(ratio.value(), core_square.value());
  l = make_cuDoubleComplex(cuCreal(l) + cusp[b * planes + 1 + 3 * targets + e], cuCimag(l));
  const auto r = physical_square.value();
  lap_log[b * output_electrons + e] = l;
  lap_ratio[b * output_electrons + e] = make_cuDoubleComplex(cuCreal(l) + cuCreal(r), cuCimag(l) + cuCimag(r));
  if (!finite(l) || !finite(lap_ratio[b * output_electrons + e]))
    fail(status, b);
}

/// Return launch status without a host fence or taking publication authority.
Error launched()
{
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}
} // namespace

/// Slab private scratch on the caller queue; final owner status admission remains mandatory.
Error launchComplexSpatialTracesConcurrent(Stream stream, std::size_t batch, std::size_t determinants,
    std::size_t n, std::size_t targets, std::size_t output_electrons, std::size_t concurrent_targets,
    const ComplexValueChannel* channels, const ComplexValueResult* amplitude,
    const cuDoubleComplex* inverse, const cuDoubleComplex* head, const double* cusp,
    cuDoubleComplex* product, cuDoubleComplex* channel_terms, cuDoubleComplex* gradient,
    cuDoubleComplex* lap_log, cuDoubleComplex* lap_ratio, std::uint64_t* status)
{
  const auto plan = planComplexSpatialTraceWorkspace(batch, determinants, n, targets, concurrent_targets);
  if (!plan.target_capacity || output_electrons < targets || output_electrons > n || !channels || !amplitude ||
      !inverse || !head || !cusp || !product || !channel_terms || !gradient || !lap_log || !lap_ratio || !status)
    throw std::invalid_argument("Complex concurrent spatial traces require an admitted owner and valid spans");
  const auto mul = [](auto a, auto b) {
    return spatial_detail::checkedProduct(a, b, "Complex concurrent spatial trace input extent overflow");
  };
  const auto planes = spatial_detail::checkedAdd(std::size_t(1), mul(std::size_t(4), targets),
      "Complex concurrent spatial trace planes overflow");
  (void)mul(planes, mul(mul(batch, determinants), mul(n, n)));
  (void)mul(std::size_t(3), mul(batch, output_electrons));
  (void)mul(batch, planes);
  for (std::size_t begin = 0; begin < targets;)
  {
    const auto live = std::min(plan.target_capacity, targets - begin);
    channelKernel<<<static_cast<unsigned int>(batch * live * determinants), 128, 0, stream>>>(batch,
        determinants, n, targets, begin, live, plan.target_capacity, inverse, head, product, channel_terms, status);
    auto result = launched();
    if (result != success)
      return result;
    combineKernel<<<static_cast<unsigned int>(batch * live), 32, 0, stream>>>(determinants, targets,
        output_electrons, begin, live, plan.target_capacity, channels, amplitude, channel_terms, cusp,
        gradient, lap_log, lap_ratio, status);
    result = launched();
    if (result != success)
      return result;
    begin += live;
  }
  return success;
}
} // namespace qmcplusplus::psiformer::device
#undef QMC_PF_TRACE_MUL
#undef QMC_PF_TRACE_DIV
#undef QMC_PF_TRACE_SUB
