//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include "PsiFormerDeviceComplexSpatialKernels.h"
#include "PsiFormerDeviceComplexWeightedProducts.h"
#include <limits>
#include <cfloat>
#include <cstdlib>
#include <stdexcept>

// Keep unmapped vendor arithmetic names private to this translation unit.
#ifdef QMC_CUDA2HIP
#define QMC_PF_MUL hipCmul
#define QMC_PF_DIV hipCdiv
#define QMC_PF_SUB hipCsub
#else
#define QMC_PF_MUL cuCmul
#define QMC_PF_DIV cuCdiv
#define QMC_PF_SUB cuCsub
#endif

namespace qmcplusplus::psiformer::device
{
namespace
{
/// Compensate Cartesian components independently, never a Hermitian product.
struct ComplexSum
{
  device_determinant::CompensatedSumBinary64 re, im;

  /// Add one ordinary complex term in a fixed order.
  __device__ void add(cuDoubleComplex z)
  {
    re.add(cuCreal(z));
    im.add(cuCimag(z));
  }

  /// Return the compensated complex total.
  __device__ cuDoubleComplex value() const { return make_cuDoubleComplex(re.value(), im.value()); }
};

/// Use release-safe component admission for every published or intermediate result.
__device__ bool finite(cuDoubleComplex z)
{ return device_math::isFiniteBinary64(cuCreal(z)) && device_math::isFiniteBinary64(cuCimag(z)); }

/// Record failure before the owner may publish any logical configuration.
__device__ void fail(std::uint64_t* status, std::size_t index)
{ atomicMin(reinterpret_cast<unsigned long long*>(status), static_cast<unsigned long long>((std::uint64_t(index) << 8) | 1)); }

/// Apply the real-envelope product rule to one Cartesian head component.
__device__ double component(const double* e, const double* z, std::size_t plane, std::size_t element,
    std::size_t matrix_span, std::size_t targets)
{
  const double value = e[element] * z[plane * matrix_span + element];
  if (plane == 0)
    return value;
  double result = value + e[plane * matrix_span + element] * z[element];
  // Compact plane classification uses K, not the physical matrix dimension N.
  if (plane > 3 * targets)
    for (std::size_t mu = 0; mu < 3; ++mu)
    {
      const auto gradient_plane = 1 + 3 * (plane - 1 - 3 * targets) + mu;
      result += 2 * e[gradient_plane * matrix_span + element] * z[gradient_plane * matrix_span + element];
    }
  return result;
}

/// Pack once: contiguous value matrices become LU while derivative planes survive.
__global__ void headKernel(OpenOrbitalJetLayout layout, const double* envelope, const double* real,
    const double* imag, cuDoubleComplex* head, std::uint64_t* status)
{
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto matrix_span = layout.orbitals.element_count;
  const auto m = layout.orbitals.configuration_count * matrix_span, planes = layout.orbitals.plane_count;
  if (index >= m * planes)
    return;
  const auto plane = index / m, configuration = (index % m) / matrix_span, element = index % matrix_span;
  const auto origin = configuration * planes * matrix_span;
  head[index] = make_cuDoubleComplex(component(envelope + origin, real + origin, plane, element,
      matrix_span, layout.orbitals.laplacian_lanes), component(envelope + origin, imag + origin, plane,
      element, matrix_span, layout.orbitals.laplacian_lanes));
  if (!finite(head[index]))
    fail(status, index);
}

/// Compute tr(A^-1 dA), with an ordinary transposed index pairing.
__device__ cuDoubleComplex trace(const cuDoubleComplex* inverse, const cuDoubleComplex* derivative, std::size_t n)
{
  ComplexSum sum;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      sum.add(QMC_PF_MUL(inverse[i * n + j], derivative[j * n + i]));
  return sum.value();
}

/// Reuse one N*N product, summing tr((A^-1 dA)^2), not its squared norm.
__device__ cuDoubleComplex squareTrace(const cuDoubleComplex* inverse, const cuDoubleComplex* derivative,
    std::size_t n, cuDoubleComplex* product)
{
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
    {
      ComplexSum sum;
      for (std::size_t k = 0; k < n; ++k)
        sum.add(QMC_PF_MUL(inverse[i * n + k], derivative[k * n + j]));
      product[i * n + j] = sum.value();
    }
  ComplexSum sum;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      sum.add(QMC_PF_MUL(product[i * n + j], product[j * n + i]));
  return sum.value();
}

/// Strict full-population inverse admission, determinant traces, then real cusp composition.
__global__ void tracesKernel(std::size_t batch, std::size_t determinants, std::size_t n, std::size_t targets,
    const ComplexValueChannel* channels, const ComplexValueResult* amplitude, const cuDoubleComplex* inverse,
    const cuDoubleComplex* head, const double* cusp, cuDoubleComplex* product,
    cuDoubleComplex* gradient, cuDoubleComplex* lap_log, cuDoubleComplex* lap_ratio, std::uint64_t* status)
{
  const auto b = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= batch)
    return;
  const auto nn = n * n, matrix_span = determinants * nn, m = batch * matrix_span, planes = 1 + 4 * targets;
  const auto* records = channels + b * determinants;
  double shift = -DBL_MAX;
  // Every canonical coefficient is1: singular channels cannot be skipped at zero weight.
  for (std::size_t d = 0; d < determinants; ++d)
  {
    if (records[d].status != 0 || amplitude[b].status != 0 || !finite(records[d].phase) ||
        !device_math::isFiniteBinary64(records[d].log_abs) || !finite(amplitude[b].phase) ||
        !device_math::isFiniteBinary64(amplitude[b].log_abs) ||
        fmax(fabs(cuCreal(amplitude[b].phase)), fabs(cuCimag(amplitude[b].phase))) == 0)
    {
      fail(status, b);
      return;
    }
    shift = fmax(shift, records[d].log_abs);
  }
  ComplexSum total;
  for (std::size_t d = 0; d < determinants; ++d)
  {
    const auto scale = exp(records[d].log_abs - shift);
    total.add(make_cuDoubleComplex(scale * cuCreal(records[d].phase), scale * cuCimag(records[d].phase)));
  }
  // Normalize without overflowing a finite-component norm or losing a subnormal denominator.
  const auto denominator = total.value();
  const auto scale = fmax(fabs(cuCreal(denominator)), fabs(cuCimag(denominator)));
  if (!finite(denominator) || !(scale > 0))
  {
    fail(status, b);
    return;
  }
  const auto re = cuCreal(denominator) / scale, im = cuCimag(denominator) / scale;
  const auto norm = hypot(re, im), log_total = log(scale) + log(norm);
  const auto phase = make_cuDoubleComplex(re / norm, im / norm);
  if (!device_math::isFiniteBinary64(log_total) || !finite(phase))
  {
    fail(status, b);
    return;
  }
  for (std::size_t lane = 0; lane < 3 * targets; ++lane)
  {
    ComplexSum sum;
    for (std::size_t d = 0; d < determinants; ++d)
    {
      const auto term = trace(inverse + b * matrix_span + d * nn,
          head + (1 + lane) * m + b * matrix_span + d * nn, n);
      cuDoubleComplex weighted;
      // Tiny weights cannot admit an unweighted overflow or erase a finite weighted trace.
      if (!finite(term) || !complex_detail::weightedProduct((records[d].log_abs - shift) - log_total,
          QMC_PF_DIV(records[d].phase, phase), term, weighted))
      {
        fail(status, b);
        return;
      }
      sum.add(weighted);
    }
    gradient[3 * b * targets + lane] = sum.value();
  }
  for (std::size_t e = 0; e < targets; ++e)
  {
    ComplexSum ratio;
    for (std::size_t d = 0; d < determinants; ++d)
    {
      const auto offset = b * matrix_span + d * nn;
      ComplexSum channel;
      channel.add(trace(inverse + offset, head + (1 + 3 * targets + e) * m + offset, n));
      for (std::size_t mu = 0; mu < 3; ++mu)
      {
        const auto* derivative = head + (1 + 3 * e + mu) * m + offset;
        const auto t = trace(inverse + offset, derivative, n);
        channel.add(QMC_PF_SUB(QMC_PF_MUL(t, t), squareTrace(inverse + offset, derivative, n, product + b * nn)));
      }
      cuDoubleComplex weighted;
      // Preserve the original finite unweighted quadratic domain before stable weighting.
      if (!finite(channel.value()) || !complex_detail::weightedProduct((records[d].log_abs - shift) - log_total,
          QMC_PF_DIV(records[d].phase, phase), channel.value(), weighted))
      {
        fail(status, b);
        return;
      }
      ratio.add(weighted);
    }
    // Ordinary complex squares retain the phase-gradient kinetic cross terms.
    ComplexSum core_square, physical_square;
    for (std::size_t mu = 0; mu < 3; ++mu)
    {
      auto& g = gradient[3 * (b * targets + e) + mu];
      core_square.add(QMC_PF_MUL(g, g));
      g = make_cuDoubleComplex(cuCreal(g) + cusp[b * planes + 1 + 3 * e + mu], cuCimag(g));
      physical_square.add(QMC_PF_MUL(g, g));
      if (!finite(g))
        fail(status, b);
    }
    auto l = QMC_PF_SUB(ratio.value(), core_square.value());
    l = make_cuDoubleComplex(cuCreal(l) + cusp[b * planes + 1 + 3 * targets + e], cuCimag(l));
    const auto r = physical_square.value();
    lap_log[b * targets + e] = l;
    lap_ratio[b * targets + e] = make_cuDoubleComplex(cuCreal(l) + cuCreal(r), cuCimag(l) + cuCimag(r));
    if (!finite(l) || !finite(lap_ratio[b * targets + e]))
      fail(status, b);
  }
}

/// Preserve the scalar channel normalization before admitting a cooperative block.
__device__ bool cooperativeSpatialCombination(std::size_t determinants,
    const ComplexValueChannel* records, const ComplexValueResult& amplitude,
    double& shift, double& log_total, cuDoubleComplex& phase)
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

/// Parallel independent complex product entries; ordinary signed traces keep scalar order.
__global__ void tracesCooperativeKernel(std::size_t batch, std::size_t determinants,
    std::size_t n, std::size_t targets, const ComplexValueChannel* channels,
    const ComplexValueResult* amplitude, const cuDoubleComplex* inverse,
    const cuDoubleComplex* head, const double* cusp, cuDoubleComplex* product,
    cuDoubleComplex* gradient, cuDoubleComplex* lap_log, cuDoubleComplex* lap_ratio,
    std::uint64_t* status)
{
  const auto b = std::size_t(blockIdx.x);
  const auto nn = n * n, matrix_span = determinants * nn, m = batch * matrix_span, planes = 1 + 4 * targets;
  const auto* records = channels + b * determinants;
  auto* scratch = product + b * nn;
  __shared__ double shift, log_total;
  __shared__ cuDoubleComplex phase;
  __shared__ int rejected;
  if (threadIdx.x == 0)
  {
    rejected = !cooperativeSpatialCombination(determinants, records, amplitude[b], shift, log_total, phase);
    if (rejected)
      fail(status, b);
  }
  __syncthreads();
  if (rejected)
    return;

  if (threadIdx.x == 0)
    for (std::size_t lane = 0; lane < 3 * targets; ++lane)
    {
      ComplexSum sum;
      for (std::size_t d = 0; d < determinants; ++d)
      {
        const auto term = trace(inverse + b * matrix_span + d * nn,
            head + (1 + lane) * m + b * matrix_span + d * nn, n);
        cuDoubleComplex weighted;
        if (!finite(term) || !complex_detail::weightedProduct((records[d].log_abs - shift) - log_total,
            QMC_PF_DIV(records[d].phase, phase), term, weighted))
        {
          fail(status, b);
          rejected = 1;
          break;
        }
        sum.add(weighted);
      }
      if (rejected)
        break;
      gradient[3 * b * targets + lane] = sum.value();
    }
  __syncthreads();
  if (rejected)
    return;

  for (std::size_t e = 0; e < targets; ++e)
  {
    ComplexSum ratio;
    for (std::size_t d = 0; d < determinants; ++d)
    {
      const auto offset = b * matrix_span + d * nn;
      const auto* matrix_inverse = inverse + offset;
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
            sum.add(QMC_PF_MUL(matrix_inverse[row * n + k], derivative[k * n + column]));
          scratch[entry] = sum.value();
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
          const auto t = trace(matrix_inverse, derivative, n);
          channel.add(QMC_PF_SUB(QMC_PF_MUL(t, t), trace(scratch, scratch, n)));
        }
        // Finish the signed-square read before another direction overwrites it.
        __syncthreads();
      }
      if (threadIdx.x == 0)
      {
        cuDoubleComplex weighted;
        if (!finite(channel.value()) || !complex_detail::weightedProduct((records[d].log_abs - shift) - log_total,
            QMC_PF_DIV(records[d].phase, phase), channel.value(), weighted))
        {
          fail(status, b);
          rejected = 1;
        }
        else
          ratio.add(weighted);
      }
      __syncthreads();
      if (rejected)
        return;
    }
    if (threadIdx.x == 0)
    {
      ComplexSum core_square, physical_square;
      for (std::size_t mu = 0; mu < 3; ++mu)
      {
        auto& g = gradient[3 * (b * targets + e) + mu];
        core_square.add(QMC_PF_MUL(g, g));
        g = make_cuDoubleComplex(cuCreal(g) + cusp[b * planes + 1 + 3 * e + mu], cuCimag(g));
        physical_square.add(QMC_PF_MUL(g, g));
        if (!finite(g))
          fail(status, b);
      }
      auto l = QMC_PF_SUB(ratio.value(), core_square.value());
      l = make_cuDoubleComplex(cuCreal(l) + cusp[b * planes + 1 + 3 * targets + e], cuCimag(l));
      const auto r = physical_square.value();
      lap_log[b * targets + e] = l;
      lap_ratio[b * targets + e] = make_cuDoubleComplex(cuCreal(l) + cuCreal(r), cuCimag(l) + cuCimag(r));
      if (!finite(l) || !finite(lap_ratio[b * targets + e]))
        fail(status, b);
    }
    __syncthreads();
  }
}

/// Preserve the scalar default unless the cooperative trace experiment is selected.
bool cooperativeSpatialTracesRequested()
{
  const char* option = std::getenv("PSIFORMER_GPU_COOPERATIVE_SPATIAL_TRACES");
  return option && option[0] == '1' && option[1] == '\0';
}

/// Check launch/index admission independently of the owner's checked preflight.
unsigned blocks(std::size_t count)
{
  if (!count || (count - 1) / 128 >= std::size_t(std::numeric_limits<int>::max()) ||
      count - 1 > (std::numeric_limits<std::uint64_t>::max() >> 8))
    throw std::length_error("Complex FULL spatial launch extent is invalid");
  return static_cast<unsigned>((count - 1) / 128 + 1);
}

/// Return the vendor launch status without hiding the owner's status synchronization.
Error launched()
{
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}
} // namespace

/// Pack product-rule jets once without a second complex head or LU copy.
Error launchComplexSpatialHead(Stream stream, OpenOrbitalJetLayout layout, const double* envelope,
    const double* real, const double* imag, cuDoubleComplex* head, std::uint64_t* status)
{
  validateOpenOrbitalJetLayout(layout);
  if (layout.features.mode != SpatialJetMode::FULL_VGL || !envelope || !real || !imag || !head || !status)
    throw std::invalid_argument("Complex FULL head inputs are invalid");
  if (layout.orbitals.plane_stride != layout.orbitals.element_count ||
      layout.orbitals.configuration_stride != layout.orbitals.plane_count * layout.orbitals.element_count)
    throw std::invalid_argument("Complex FULL head requires packed real source planes");
  headKernel<<<blocks(spatialJetSpanElements(layout.orbitals)), 128, 0, stream>>>(layout, envelope, real, imag, head, status);
  return launched();
}

/// Consume the admitted head derivatives and inverses immediately into complete outputs.
Error launchComplexSpatialTraces(Stream stream, std::size_t batch, std::size_t determinants, std::size_t n,
    const ComplexValueChannel* channels, const ComplexValueResult* amplitude, const cuDoubleComplex* inverse,
    const cuDoubleComplex* head, const double* cusp, cuDoubleComplex* product,
    cuDoubleComplex* gradient, cuDoubleComplex* lap_log, cuDoubleComplex* lap_ratio, std::uint64_t* status)
{
  if (!determinants || !n || !channels || !amplitude || !inverse || !head || !cusp || !product ||
      !gradient || !lap_log || !lap_ratio || !status)
    throw std::invalid_argument("Complex FULL trace inputs are invalid");
  const auto block_count = blocks(batch);
  if (cooperativeSpatialTracesRequested() && batch <= std::numeric_limits<int>::max())
    tracesCooperativeKernel<<<static_cast<unsigned int>(batch), 128, 0, stream>>>(
        batch, determinants, n, n, channels, amplitude, inverse, head, cusp, product, gradient, lap_log, lap_ratio, status);
  else
    tracesKernel<<<block_count, 128, 0, stream>>>(batch, determinants, n, n, channels, amplitude,
        inverse, head, cusp, product, gradient, lap_log, lap_ratio, status);
  return launched();
}

/// Reuse the SAME contraction for one compact physical-target tape; FULL API stays unchanged.
Error launchComplexSpatialTraces(Stream stream, OpenOrbitalJetLayout layout,
    const ComplexValueChannel* channels, const ComplexValueResult* amplitude, const cuDoubleComplex* inverse,
    const cuDoubleComplex* head, const double* cusp, cuDoubleComplex* product,
    cuDoubleComplex* gradient, cuDoubleComplex* lap_log, cuDoubleComplex* lap_ratio, std::uint64_t* status)
{
  validateOpenOrbitalJetLayout(layout);
  const auto n = layout.features.electron_count;
  const auto nn = spatial_detail::checkedProduct(n, n, "Complex target trace extent overflow");
  const auto m = spatial_detail::checkedProduct(layout.determinant_count, nn, "Complex target trace extent overflow");
  if (layout.features.configuration_count != 1 || layout.features.mode != SpatialJetMode::FULL_VGL ||
      layout.orbitals.element_count != m || layout.orbital_row_stride != n || layout.orbital_determinant_stride != nn ||
      layout.orbitals.plane_stride != m || layout.orbitals.configuration_stride !=
          spatial_detail::checkedProduct(layout.orbitals.plane_count, m, "Complex target trace extent overflow") ||
      !channels || !amplitude || !inverse ||
      !head || !cusp || !product || !gradient || !lap_log || !lap_ratio || !status)
    throw std::invalid_argument("Complex target traces require packed B1 current jets");
  if (cooperativeSpatialTracesRequested())
    tracesCooperativeKernel<<<1, 128, 0, stream>>>(1, layout.determinant_count, layout.features.electron_count,
        layout.orbitals.laplacian_lanes, channels, amplitude, inverse, head, cusp, product, gradient, lap_log, lap_ratio, status);
  else
    tracesKernel<<<1, 128, 0, stream>>>(1, layout.determinant_count, layout.features.electron_count,
        layout.orbitals.laplacian_lanes, channels, amplitude, inverse, head, cusp, product, gradient, lap_log, lap_ratio, status);
  return launched();
}
} // namespace qmcplusplus::psiformer::device
#undef QMC_PF_MUL
#undef QMC_PF_DIV
#undef QMC_PF_SUB
