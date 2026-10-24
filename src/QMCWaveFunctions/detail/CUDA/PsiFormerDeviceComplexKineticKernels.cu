//////////////////////////////////////////////////////////////////////////////////////
// Distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory. Copyright (c) 2026 QMCPACK developers.
//////////////////////////////////////////////////////////////////////////////////////
#include "PsiFormerDeviceComplexKineticKernels.h"
#include "PsiFormerDeviceComplexWeightedProducts.h"
#include <cfloat>
#include <limits>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// Keep vendor arithmetic selection local; all products below are ordinary complex.
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
using C = cuDoubleComplex;

/// Componentwise finite admission avoids overflowing a representable complex norm.
__device__ bool finite(C z)
{ return device_math::isFiniteBinary64(cuCreal(z)) && device_math::isFiniteBinary64(cuCimag(z)); }

/// Cartesian addition preserves ordinary, not conjugate, algebra.
__device__ C add(C a, C b)
{ return make_cuDoubleComplex(cuCreal(a) + cuCreal(b), cuCimag(a) + cuCimag(b)); }

/// Multiply by a real scalar without introducing a complex norm.
__device__ C scale(C a, double s)
{ return make_cuDoubleComplex(s * cuCreal(a), s * cuCimag(a)); }

/// Compensate real and imaginary sums independently in deterministic order.
struct Sum
{
  device_determinant::CompensatedSumBinary64 re, im;

  /// Add one ordinary complex contribution in Cartesian order.
  __device__ void push(C z)
  {
    re.add(cuCreal(z));
    im.add(cuCimag(z));
  }

  /// Recover the two corrected components without forming a norm.
  __device__ C value() const { return make_cuDoubleComplex(re.value(), im.value()); }
};

/// Failure is a numerical admission error, never a node or a Monte Carlo rejection.
__device__ void fail(std::uint64_t* status)
{ atomicMin(reinterpret_cast<unsigned long long*>(status), 1ULL); }

/// Explicit finite-factor domain: admit every unweighted product and matrix sum.
__device__ bool multiply(const C* left, const C* right, C* output, std::size_t n)
{
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
    {
      Sum sum;
      for (std::size_t k = 0; k < n; ++k)
      {
        const auto term = QMC_PF_MUL(left[i * n + k], right[k * n + j]);
        if (!finite(term)) return false;
        sum.push(term);
      }
      output[i * n + j] = sum.value();
      if (!finite(output[i * n + j])) return false;
    }
  return true;
}

/// Trace pairs diagonal entries without conjugating either factor.
__device__ C trace(const C* matrix, std::size_t n)
{
  Sum sum;
  for (std::size_t i = 0; i < n; ++i) sum.push(matrix[i * n + i]);
  return sum.value();
}

/// Apply log-weight only AFTER finite combined-term admission, then add to a seed.
__device__ bool weightedAdd(double log_weight, C phase, C term, C& output)
{
  C weighted;
  if (!finite(term) || !complex_detail::weightedProduct(log_weight, phase, term, weighted)) return false;
  output = add(output, weighted);
  return finite(output);
}

/// CPU-qualified matrix-root formula, with compact physical target planes and original I.
__global__ void matrixRoots(OpenOrbitalJetLayout layout, const ComplexValueChannel* records,
    const ComplexValueResult* amplitude, const C* inverse, const C* head, const C* gradient,
    const C* laplacian, const C* qg, const C* ql, C* output, C* scratch, std::uint64_t* status)
{
  const auto n = layout.features.electron_count, nn = n * n, d = layout.determinant_count;
  const auto m = layout.orbitals.element_count, k = layout.orbitals.laplacian_lanes;
  if (amplitude->status || !finite(amplitude->phase) || !device_math::isFiniteBinary64(amplitude->log_abs) ||
      fmax(fabs(cuCreal(amplitude->phase)), fabs(cuCimag(amplitude->phase))) == 0)
  { fail(status); return; }
  double shift = -DBL_MAX;
  for (std::size_t channel = 0; channel < d; ++channel)
  {
    if (records[channel].status || !finite(records[channel].phase) ||
        fmax(fabs(cuCreal(records[channel].phase)), fabs(cuCimag(records[channel].phase))) == 0 ||
        !device_math::isFiniteBinary64(records[channel].log_abs)) { fail(status); return; }
    shift = fmax(shift, records[channel].log_abs);
  }
  for (std::size_t i = 0; i < m; ++i)
    if (!finite(inverse[i])) { fail(status); return; }
  for (std::size_t i = 0; i < 3 * k; ++i)
    if (!finite(gradient[i]) || !finite(qg[i])) { fail(status); return; }
  for (std::size_t i = 0; i < k; ++i)
    if (!finite(laplacian[i]) || !finite(ql[i])) { fail(status); return; }
  Sum denominator;
  for (std::size_t channel = 0; channel < d; ++channel)
    denominator.push(scale(records[channel].phase, exp(records[channel].log_abs - shift)));
  const auto total = denominator.value();
  const auto maximum = fmax(fabs(cuCreal(total)), fabs(cuCimag(total)));
  if (!finite(total) || !(maximum > 0)) { fail(status); return; }
  const auto re = cuCreal(total) / maximum, im = cuCimag(total) / maximum, norm = hypot(re, im);
  const auto phase = make_cuDoubleComplex(re / norm, im / norm);
  const auto log_total = log(maximum) + log(norm);
  if (!finite(phase) || !device_math::isFiniteBinary64(log_total)) { fail(status); return; }
  for (std::size_t i = 0; i < layout.orbitals.plane_count * m; ++i) output[i] = make_cuDoubleComplex(0, 0);

  C* a = scratch;
  C* b = a + nn;
  C* c = b + nn;
  for (std::size_t channel = 0; channel < d; ++channel)
  {
    const auto begin = channel * nn;
    const auto* inv = inverse + begin;
    const auto log_weight = (records[channel].log_abs - shift) - log_total;
    const auto ratio = QMC_PF_DIV(records[channel].phase, phase);
    // No channel is skipped, even if exp(log_weight) would underflow.
    for (std::size_t e = 0; e < k; ++e)
    {
      auto alpha = scale(laplacian[e], -1);
      for (std::size_t mu = 0; mu < 3; ++mu)
      {
        const auto lane = 3 * e + mu;
        if (!multiply(inv, head + (1 + lane) * m + begin, a, n)) { fail(status); return; }
        const auto delta = add(trace(a, n), scale(gradient[lane], -1));
        const auto square = QMC_PF_MUL(delta, delta);
        if (!finite(delta) || !finite(square)) { fail(status); return; }
        alpha = add(alpha, square);
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = 0; j < n; ++j)
          {
            const auto term = QMC_PF_MUL(a[i * n + j], a[j * n + i]);
            if (!finite(term)) { fail(status); return; }
            alpha = add(alpha, scale(term, -1));
          }
        if (!finite(alpha) || !multiply(a, inv, b, n) || !multiply(a, b, c, n)) { fail(status); return; }
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = 0; j < n; ++j)
          {
            const auto index = begin + i * n + j, t = j * n + i;
            const auto delta_i = QMC_PF_MUL(delta, inv[t]), delta_b = QMC_PF_MUL(delta, b[t]);
            const auto combined = add(delta_i, scale(b[t], -1)), second = add(c[t], scale(delta_b, -1));
            const auto value_g = QMC_PF_MUL(qg[lane], combined), value_l = QMC_PF_MUL(ql[e], second);
            const auto grad_g = QMC_PF_MUL(qg[lane], inv[t]), grad_l = QMC_PF_MUL(ql[e], combined);
            if (!finite(delta_i) || !finite(delta_b) || !finite(combined) || !finite(second) ||
                !finite(value_g) || !finite(value_l) || !finite(grad_g) || !finite(grad_l) ||
                !weightedAdd(log_weight, ratio, add(value_g, scale(value_l, 2)), output[index]) ||
                !weightedAdd(log_weight, ratio, add(grad_g, scale(grad_l, 2)), output[(1 + lane) * m + index]))
            { fail(status); return; }
          }
      }
      if (!multiply(inv, head + (1 + 3 * k + e) * m + begin, a, n)) { fail(status); return; }
      alpha = add(alpha, trace(a, n));
      if (!finite(alpha) || !multiply(a, inv, b, n)) { fail(status); return; }
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
        {
          const auto index = begin + i * n + j, t = j * n + i;
          const auto factor = QMC_PF_MUL(alpha, inv[t]);
          const auto combined = add(factor, scale(b[t], -1));
          if (!finite(factor) || !finite(combined) ||
              !weightedAdd(log_weight, ratio, QMC_PF_MUL(ql[e], combined), output[index]) ||
              !weightedAdd(log_weight, ratio, QMC_PF_MUL(ql[e], inv[t]), output[(1 + 3 * k + e) * m + index]))
          { fail(status); return; }
        }
    }
  }
}

/// Admit the same scalar channel/physical-root domain before any cooperative mutation.
__device__ bool rootCombination(std::size_t d, std::size_t m, std::size_t k,
    const ComplexValueChannel* records, const ComplexValueResult* amplitude, const C* inverse,
    const C* gradient, const C* laplacian, const C* qg, const C* ql,
    double& shift, C& phase, double& log_total)
{
  if (amplitude->status || !finite(amplitude->phase) || !device_math::isFiniteBinary64(amplitude->log_abs) ||
      fmax(fabs(cuCreal(amplitude->phase)), fabs(cuCimag(amplitude->phase))) == 0)
    return false;
  shift = -DBL_MAX;
  for (std::size_t channel = 0; channel < d; ++channel)
  {
    if (records[channel].status || !finite(records[channel].phase) ||
        fmax(fabs(cuCreal(records[channel].phase)), fabs(cuCimag(records[channel].phase))) == 0 ||
        !device_math::isFiniteBinary64(records[channel].log_abs))
      return false;
    shift = fmax(shift, records[channel].log_abs);
  }
  for (std::size_t i = 0; i < m; ++i)
    if (!finite(inverse[i]))
      return false;
  for (std::size_t i = 0; i < 3 * k; ++i)
    if (!finite(gradient[i]) || !finite(qg[i]))
      return false;
  for (std::size_t i = 0; i < k; ++i)
    if (!finite(laplacian[i]) || !finite(ql[i]))
      return false;
  Sum denominator;
  for (std::size_t channel = 0; channel < d; ++channel)
    denominator.push(scale(records[channel].phase, exp(records[channel].log_abs - shift)));
  const auto total = denominator.value();
  const auto maximum = fmax(fabs(cuCreal(total)), fabs(cuCimag(total)));
  if (!finite(total) || !(maximum > 0))
    return false;
  const auto re = cuCreal(total) / maximum, im = cuCimag(total) / maximum, norm = hypot(re, im);
  phase = make_cuDoubleComplex(re / norm, im / norm);
  log_total = log(maximum) + log(norm);
  return finite(phase) && device_math::isFiniteBinary64(log_total);
}

/// Snapshot failure uniformly before any fast lane may mutate the next stage's flag.
__device__ bool rootStageRejected(const unsigned& rejected)
{
  __syncthreads();
  const bool failed = rejected != 0;
  __syncthreads();
  return failed;
}

/// Independent output entries retain the old increasing-k compensated Cartesian sum.
/// All threads join before returning the shared failure state to the caller.
__device__ bool multiplyCooperative(const C* left, const C* right, C* output,
    std::size_t n, unsigned& rejected)
{
  for (auto index = std::size_t(threadIdx.x); index < n * n; index += blockDim.x)
  {
    const auto i = index / n, j = index % n;
    Sum sum;
    for (std::size_t k = 0; k < n; ++k)
    {
      const auto term = QMC_PF_MUL(left[i * n + k], right[k * n + j]);
      if (!finite(term))
      {
        atomicOr(&rejected, 1u);
        break;
      }
      sum.push(term);
    }
    output[index] = sum.value();
    if (!finite(output[index]))
      atomicOr(&rejected, 1u);
  }
  return !rootStageRejected(rejected);
}

/// Each selected channel owns three N*N matrices; canonical replay preserves failed prefixes.
template<bool ChannelParallel = false, bool FailedOnly = false>
__global__ void matrixRootsCooperative(OpenOrbitalJetLayout layout, const ComplexValueChannel* records,
    const ComplexValueResult* amplitude, const C* inverse, const C* head, const C* gradient,
    const C* laplacian, const C* qg, const C* ql, C* output, C* scratch, std::uint64_t* status,
    const std::uint64_t* replay_failure)
{
  if constexpr (FailedOnly)
    if (*replay_failure == ~std::uint64_t(0))
      return;
  const auto n = layout.features.electron_count, nn = n * n, d = layout.determinant_count;
  const auto m = layout.orbitals.element_count, k = layout.orbitals.laplacian_lanes;
  const auto first = ChannelParallel ? std::size_t(blockIdx.x) : 0;
  const auto last = ChannelParallel ? first + 1 : d;
  __shared__ unsigned rejected;
  __shared__ double shift, log_total, log_weight;
  __shared__ C phase, ratio, alpha, delta;
  if (threadIdx.x == 0)
    rejected = !rootCombination(d, m, k, records, amplitude, inverse, gradient, laplacian, qg, ql,
        shift, phase, log_total);
  if (rootStageRejected(rejected))
  {
    if (threadIdx.x == 0) fail(status);
    return;
  }
  if constexpr (ChannelParallel)
  {
    for (auto i = std::size_t(threadIdx.x); i < layout.orbitals.plane_count * nn; i += blockDim.x)
      output[(i / nn) * m + first * nn + i % nn] = make_cuDoubleComplex(0, 0);
  }
  else
    for (auto i = std::size_t(threadIdx.x); i < layout.orbitals.plane_count * m; i += blockDim.x)
      output[i] = make_cuDoubleComplex(0, 0);
  __syncthreads();
  C* a = scratch + (ChannelParallel ? 3 * first * nn : 0);
  C* b = a + nn;
  C* c = b + nn;
  for (std::size_t channel = first; channel < last; ++channel)
  {
    const auto begin = channel * nn;
    const auto* inv = inverse + begin;
    if (threadIdx.x == 0)
    {
      log_weight = (records[channel].log_abs - shift) - log_total;
      ratio = QMC_PF_DIV(records[channel].phase, phase);
    }
    __syncthreads();
    for (std::size_t e = 0; e < k; ++e)
    {
      if (threadIdx.x == 0) alpha = scale(laplacian[e], -1);
      for (std::size_t mu = 0; mu < 3; ++mu)
      {
        const auto lane = 3 * e + mu;
        if (!multiplyCooperative(inv, head + (1 + lane) * m + begin, a, n, rejected))
        {
          if (threadIdx.x == 0) fail(status);
          return;
        }
        if (threadIdx.x == 0)
        {
          delta = add(trace(a, n), scale(gradient[lane], -1));
          const auto square = QMC_PF_MUL(delta, delta);
          if (!finite(delta) || !finite(square)) rejected = 1;
          alpha = add(alpha, square);
          for (std::size_t i = 0; i < n && !rejected; ++i)
            for (std::size_t j = 0; j < n; ++j)
            {
              const auto term = QMC_PF_MUL(a[i * n + j], a[j * n + i]);
              if (!finite(term))
              {
                rejected = 1;
                break;
              }
              alpha = add(alpha, scale(term, -1));
            }
          if (!finite(alpha)) rejected = 1;
        }
        if (rootStageRejected(rejected) || !multiplyCooperative(a, inv, b, n, rejected) ||
            !multiplyCooperative(a, b, c, n, rejected))
        {
          if (threadIdx.x == 0) fail(status);
          return;
        }
        for (auto ij = std::size_t(threadIdx.x); ij < nn; ij += blockDim.x)
        {
          const auto i = ij / n, j = ij % n, index = begin + ij, t = j * n + i;
          const auto delta_i = QMC_PF_MUL(delta, inv[t]), delta_b = QMC_PF_MUL(delta, b[t]);
          const auto combined = add(delta_i, scale(b[t], -1)), second = add(c[t], scale(delta_b, -1));
          const auto value_g = QMC_PF_MUL(qg[lane], combined), value_l = QMC_PF_MUL(ql[e], second);
          const auto grad_g = QMC_PF_MUL(qg[lane], inv[t]), grad_l = QMC_PF_MUL(ql[e], combined);
          if (!finite(delta_i) || !finite(delta_b) || !finite(combined) || !finite(second) ||
              !finite(value_g) || !finite(value_l) || !finite(grad_g) || !finite(grad_l) ||
              !weightedAdd(log_weight, ratio, add(value_g, scale(value_l, 2)), output[index]) ||
              !weightedAdd(log_weight, ratio, add(grad_g, scale(grad_l, 2)), output[(1 + lane) * m + index]))
            atomicOr(&rejected, 1u);
        }
        if (rootStageRejected(rejected))
        {
          if (threadIdx.x == 0) fail(status);
          return;
        }
      }
      if (!multiplyCooperative(inv, head + (1 + 3 * k + e) * m + begin, a, n, rejected))
      {
        if (threadIdx.x == 0) fail(status);
        return;
      }
      if (threadIdx.x == 0)
      {
        alpha = add(alpha, trace(a, n));
        if (!finite(alpha)) rejected = 1;
      }
      if (rootStageRejected(rejected) || !multiplyCooperative(a, inv, b, n, rejected))
      {
        if (threadIdx.x == 0) fail(status);
        return;
      }
      for (auto ij = std::size_t(threadIdx.x); ij < nn; ij += blockDim.x)
      {
        const auto i = ij / n, j = ij % n, index = begin + ij, t = j * n + i;
        const auto factor = QMC_PF_MUL(alpha, inv[t]);
        const auto combined = add(factor, scale(b[t], -1));
        if (!finite(factor) || !finite(combined) ||
            !weightedAdd(log_weight, ratio, QMC_PF_MUL(ql[e], combined), output[index]) ||
            !weightedAdd(log_weight, ratio, QMC_PF_MUL(ql[e], inv[t]), output[(1 + 3 * k + e) * m + index]))
          atomicOr(&rejected, 1u);
      }
      if (rootStageRejected(rejected))
      {
        if (threadIdx.x == 0) fail(status);
        return;
      }
    }
  }
}

/// A cold opt-in affects only this matrix-root leaf, never inverse/head/trunk work.
bool cooperativeMatrixRootsRequested()
{
  static const bool requested = [] {
    const char* value = std::getenv("PSIFORMER_GPU_COOPERATIVE_COMPLEX_KINETIC_ROOTS");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::invalid_argument("PSIFORMER_GPU_COOPERATIVE_COMPLEX_KINETIC_ROOTS must be 0 or 1");
  }();
  return requested;
}

/// Select a real objective without conjugating its analytic complex cotangent.
__device__ double component(C value, std::size_t selected)
{ return selected ? cuCimag(value) : cuCreal(value); }

/// Accumulate only finite real canonical/feature contributions.
__device__ bool accumulate(double& destination, double term)
{
  if (!device_math::isFiniteBinary64(term) || !device_math::isFiniteBinary64(destination)) return false;
  destination += term;
  return device_math::isFiniteBinary64(destination);
}

/// True B*E reverse: preserve all-row VALUE terms and pi responses at zero E.
template<int Owner = 0>
__device__ void headReverseBody(ComplexKineticHeadLayout l, periodic_geometry::Cell cell,
    const double* positions, const double* nuclei, const double* p, const double* features,
    const double* envelope, const double* real, const double* imag, const C* roots,
    std::size_t selected, double* feature_seed, double* response, std::uint64_t* status,
    std::size_t owned = 0)
{
  const auto& layout = l.real.forward;
  const auto n = layout.features.electron_count, m = layout.orbitals.element_count;
  const auto u = layout.features.element_count, k = layout.orbitals.laplacian_lanes;
  // Owner 1: spin/f/d/orbital; owner 2: plane/row/f; owner 3: spin/d/orbital/atom.
  const auto width = layout.feature_width, determinants = layout.determinant_count;
  const auto orbital_owner = Owner == 3 ? owned / layout.nucleus_count % n : owned % n;
  const auto determinant_owner = Owner == 3 ? owned / layout.nucleus_count / n % determinants : owned / n % determinants;
  const auto feature_owner = Owner == 1 ? owned / n / determinants % width : owned % width;
  const auto spin_owner = Owner == 3 ? owned / layout.nucleus_count / n / determinants : owned / n / determinants / width;
  const auto row_owner = owned / width % n, plane_owner = owned / width / n;
  const auto row_begin = Owner == 2 ? row_owner :
      ((Owner == 1 || Owner == 3) && spin_owner ? layout.spin_up_count : 0);
  const auto row_end = Owner == 2 ? row_owner + 1 :
      ((Owner == 1 || Owner == 3) && !spin_owner ? layout.spin_up_count : n);
  for (std::size_t row = row_begin; row < row_end; ++row)
  {
    if constexpr (Owner == 1 || Owner == 3)
      if ((row < layout.spin_up_count ? 0u : 1u) != spin_owner) continue;
    if constexpr (Owner == 2)
      if (row != row_owner) continue;
    for (std::size_t d = (Owner == 1 || Owner == 3) ? determinant_owner : 0;
         d < ((Owner == 1 || Owner == 3) ? determinant_owner + 1 : layout.determinant_count); ++d)
      for (std::size_t orbital = (Owner == 1 || Owner == 3) ? orbital_owner : 0;
           orbital < ((Owner == 1 || Owner == 3) ? orbital_owner + 1 : n); ++orbital)
      {
        if constexpr (Owner == 1 || Owner == 3)
          if (d != determinant_owner || orbital != orbital_owner) continue;
        const bool up = row < layout.spin_up_count;
        const auto wr = up ? l.real.backflow_up_response_offset : l.real.backflow_down_response_offset;
        const auto wi = up ? l.imaginary_up : l.imaginary_down;
        const auto pi = up ? l.real.envelope_pi_up_response_offset : l.real.envelope_pi_down_response_offset;
        const auto zeta = up ? l.real.envelope_zeta_up_response_offset : l.real.envelope_zeta_down_response_offset;
        const auto element = open_orbital::outputElement(layout, d, row, orbital);
        Sum qb0, qe0;
        for (std::size_t plane = 0; plane < layout.orbitals.plane_count; ++plane)
        {
          const auto offset = plane * m + element;
          const auto b = make_cuDoubleComplex(real[offset], imag[offset]);
          const auto qb = scale(roots[offset], envelope[offset]), qe = QMC_PF_MUL(roots[offset], b);
          if (!finite(qb) || !finite(qe)) { fail(status); return; }
          qb0.push(qb); qe0.push(qe);
        }
        // Affine head and feature pullback, including cross terms of the jet product.
        if constexpr (Owner != 3)
        for (std::size_t plane = Owner == 2 ? plane_owner : 0;
             plane < (Owner == 2 ? plane_owner + 1 : layout.orbitals.plane_count); ++plane)
        {
          auto qb = plane ? scale(roots[plane * m + element], envelope[element]) : qb0.value();
          if (plane && plane <= 3 * k)
            qb = add(qb, scale(roots[(1 + 3 * k + (plane - 1) / 3) * m + element], 2 * envelope[plane * m + element]));
          if (!finite(qb)) { fail(status); return; }
          for (std::size_t f = (Owner == 1 || Owner == 2) ? feature_owner : 0;
               f < ((Owner == 1 || Owner == 2) ? feature_owner + 1 : layout.feature_width); ++f)
          {
            if constexpr (Owner == 1 || Owner == 2)
              if (f != feature_owner) continue;
            const auto index = open_orbital::weightOffset(layout, f, d, orbital);
            const auto input = plane * u + row * layout.feature_row_stride + f;
            const auto weight = make_cuDoubleComplex(p[wr + index], p[wi + index]);
            const auto head = scale(qb, features[input]), seed = QMC_PF_MUL(qb, weight);
            if (!finite(head) || !finite(seed)) { fail(status); return; }
            if constexpr (Owner != 2)
              if (!accumulate(response[wr + index], component(head, selected)) ||
                  !accumulate(response[wi + index], selected ? cuCreal(head) : -cuCimag(head)))
              { fail(status); return; }
            if constexpr (Owner != 1)
              if (!accumulate(feature_seed[input], component(seed, selected))) { fail(status); return; }
          }
        }
        // Envelope parameter jets use the established torus radial/subgradient convention.
        const bool targeted = layout.orbitals.containsTargetElectron(row);
        const auto local = targeted ? row - layout.orbitals.target_electron_begin : 0;
        const auto qlap = targeted ? roots[(1 + 3 * k + local) * m + element] : make_cuDoubleComplex(0, 0);
        const auto b0 = make_cuDoubleComplex(real[element], imag[element]);
        if constexpr (Owner == 0 || Owner == 3)
        for (std::size_t atom = Owner == 3 ? owned % layout.nucleus_count : 0;
             atom < (Owner == 3 ? owned % layout.nucleus_count + 1 : layout.nucleus_count); ++atom)
        {
          if constexpr (Owner == 3)
            if (atom != owned % layout.nucleus_count) continue;
          periodic_geometry::Pair pair;
          if (!periodic_geometry::evaluatePair(cell, positions + row * layout.position_row_stride,
              nuclei + atom * layout.nucleus_row_stride, pair, true) || !(pair.radius > 0)) { fail(status); return; }
          const auto index = open_orbital::envelopeOffset(layout, d, orbital, atom);
          const auto z = p[zeta + index], az = fabs(z), sign = z > 0 ? 1.0 : (z < 0 ? -1.0 : 0.0);
          const auto decay = exp(-az * pair.radius), weighted = p[pi + index] * decay;
          for (std::size_t direction = 0; direction < 2; ++direction)
          {
            const auto v = direction ? -sign * pair.radius * weighted : decay;
            const auto first = direction ? sign * weighted * (az * pair.radius - 1) : -az * decay;
            const auto second = direction ? sign * weighted * (-pair.radius * az * az + 2 * az) : az * az * decay;
            auto result = scale(qe0.value(), v);
            if (targeted)
            {
              result = add(result, scale(QMC_PF_MUL(qlap, b0),
                  second * pair.radius_gradient_norm_squared + first * pair.radius_laplacian));
              for (std::size_t mu = 0; mu < 3; ++mu)
              {
                const auto offset = (1 + 3 * local + mu) * m + element;
                const auto b = make_cuDoubleComplex(real[offset], imag[offset]);
                const auto term = add(QMC_PF_MUL(roots[offset], b0), scale(QMC_PF_MUL(qlap, b), 2));
                if (!finite(term)) { fail(status); return; }
                result = add(result, scale(term, first * pair.radius_gradient[mu]));
              }
            }
            if (!finite(result) || !accumulate(response[(direction ? zeta : pi) + index], component(result, selected)))
            { fail(status); return; }
          }
        }
      }
  }
}

/// Legacy single-thread entry retains canonical row/determinant/orbital/plane order.
__global__ void headReverse(ComplexKineticHeadLayout l, periodic_geometry::Cell cell,
    const double* positions, const double* nuclei, const double* p, const double* features,
    const double* envelope, const double* real, const double* imag, const C* roots,
    std::size_t selected, double* feature_seed, double* response, std::uint64_t* status)
{
  const auto& layout = l.real.forward;
  const auto n = layout.features.electron_count, m = layout.orbitals.element_count;
  const auto u = layout.features.element_count, k = layout.orbitals.laplacian_lanes;
  for (std::size_t row = 0; row < n; ++row)
    for (std::size_t d = 0; d < layout.determinant_count; ++d)
      for (std::size_t orbital = 0; orbital < n; ++orbital)
      {
        const bool up = row < layout.spin_up_count;
        const auto wr = up ? l.real.backflow_up_response_offset : l.real.backflow_down_response_offset;
        const auto wi = up ? l.imaginary_up : l.imaginary_down;
        const auto pi = up ? l.real.envelope_pi_up_response_offset : l.real.envelope_pi_down_response_offset;
        const auto zeta = up ? l.real.envelope_zeta_up_response_offset : l.real.envelope_zeta_down_response_offset;
        const auto element = open_orbital::outputElement(layout, d, row, orbital);
        Sum qb0, qe0;
        for (std::size_t plane = 0; plane < layout.orbitals.plane_count; ++plane)
        {
          const auto offset = plane * m + element;
          const auto b = make_cuDoubleComplex(real[offset], imag[offset]);
          const auto qb = scale(roots[offset], envelope[offset]), qe = QMC_PF_MUL(roots[offset], b);
          if (!finite(qb) || !finite(qe)) { fail(status); return; }
          qb0.push(qb); qe0.push(qe);
        }
        // Affine head and feature pullback, including cross terms of the jet product.
        for (std::size_t plane = 0; plane < layout.orbitals.plane_count; ++plane)
        {
          auto qb = plane ? scale(roots[plane * m + element], envelope[element]) : qb0.value();
          if (plane && plane <= 3 * k)
            qb = add(qb, scale(roots[(1 + 3 * k + (plane - 1) / 3) * m + element], 2 * envelope[plane * m + element]));
          if (!finite(qb)) { fail(status); return; }
          for (std::size_t f = 0; f < layout.feature_width; ++f)
          {
            const auto index = open_orbital::weightOffset(layout, f, d, orbital);
            const auto input = plane * u + row * layout.feature_row_stride + f;
            const auto weight = make_cuDoubleComplex(p[wr + index], p[wi + index]);
            const auto head = scale(qb, features[input]), seed = QMC_PF_MUL(qb, weight);
            if (!finite(head) || !finite(seed) || !accumulate(response[wr + index], component(head, selected)) ||
                !accumulate(response[wi + index], selected ? cuCreal(head) : -cuCimag(head)) ||
                !accumulate(feature_seed[input], component(seed, selected))) { fail(status); return; }
          }
        }
        // Envelope parameter jets use the established torus radial/subgradient convention.
        const bool targeted = layout.orbitals.containsTargetElectron(row);
        const auto local = targeted ? row - layout.orbitals.target_electron_begin : 0;
        const auto qlap = targeted ? roots[(1 + 3 * k + local) * m + element] : make_cuDoubleComplex(0, 0);
        const auto b0 = make_cuDoubleComplex(real[element], imag[element]);
        for (std::size_t atom = 0; atom < layout.nucleus_count; ++atom)
        {
          periodic_geometry::Pair pair;
          if (!periodic_geometry::evaluatePair(cell, positions + row * layout.position_row_stride,
              nuclei + atom * layout.nucleus_row_stride, pair, true) || !(pair.radius > 0)) { fail(status); return; }
          const auto index = open_orbital::envelopeOffset(layout, d, orbital, atom);
          const auto z = p[zeta + index], az = fabs(z), sign = z > 0 ? 1.0 : (z < 0 ? -1.0 : 0.0);
          const auto decay = exp(-az * pair.radius), weighted = p[pi + index] * decay;
          for (std::size_t direction = 0; direction < 2; ++direction)
          {
            const auto v = direction ? -sign * pair.radius * weighted : decay;
            const auto first = direction ? sign * weighted * (az * pair.radius - 1) : -az * decay;
            const auto second = direction ? sign * weighted * (-pair.radius * az * az + 2 * az) : az * az * decay;
            auto result = scale(qe0.value(), v);
            if (targeted)
            {
              result = add(result, scale(QMC_PF_MUL(qlap, b0),
                  second * pair.radius_gradient_norm_squared + first * pair.radius_laplacian));
              for (std::size_t mu = 0; mu < 3; ++mu)
              {
                const auto offset = (1 + 3 * local + mu) * m + element;
                const auto b = make_cuDoubleComplex(real[offset], imag[offset]);
                const auto term = add(QMC_PF_MUL(roots[offset], b0), scale(QMC_PF_MUL(qlap, b), 2));
                if (!finite(term)) { fail(status); return; }
                result = add(result, scale(term, first * pair.radius_gradient[mu]));
              }
            }
            if (!finite(result) || !accumulate(response[(direction ? zeta : pi) + index], component(result, selected)))
            { fail(status); return; }
          }
        }
      }
}

/// Copy original destination bits before any private arithmetic; initialize only private failure.
__global__ void beginHeadReverse(std::size_t p, std::size_t f, const double* response,
    const double* feature, double* private_response, double* private_feature, std::uint64_t* failure)
{
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < p) private_response[i] = response[i];
  if (i < f) private_feature[i] = feature[i];
  if (i == 0) *failure = ~std::uint64_t{0};
}

/// Each thread owns complete increasing-order reductions for disjoint output addresses.
template<int Owner>
__global__ void parallelHeadReverse(std::size_t count, ComplexKineticHeadLayout l, periodic_geometry::Cell cell,
    const double* positions, const double* nuclei, const double* p, const double* features,
    const double* envelope, const double* real, const double* imag, const C* roots,
    std::size_t selected, double* feature_seed, double* response, std::uint64_t* failure)
{
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    headReverseBody<Owner>(l, cell, positions, nuclei, p, features, envelope, real, imag, roots,
        selected, feature_seed, response, failure, i);
}

/// Stream-ordered publication or canonical failure replay never exposes private failed prefixes.
__global__ void finishHeadReverse(std::size_t pcount, std::size_t fcount, ComplexKineticHeadLayout l,
    periodic_geometry::Cell cell, const double* positions, const double* nuclei, const double* p,
    const double* features, const double* envelope, const double* real, const double* imag, const C* roots,
    std::size_t selected, double* feature_seed, double* response, std::uint64_t* status,
    const double* private_feature, const double* private_response, const std::uint64_t* failure)
{
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (*failure != ~std::uint64_t{0})
  {
    if (i == 0)
      headReverseBody(l, cell, positions, nuclei, p, features, envelope, real, imag, roots,
          selected, feature_seed, response, status);
    return;
  }
  if (i < pcount) response[i] = private_response[i];
  if (i < fcount) feature_seed[i] = private_feature[i];
}

/// Fixed physical masses/kappa generate ordinary complex roots and exactly one tile energy.
__global__ void coefficients(SpatialJetLayout layout, const C* g, const C* l, const double* cusp,
    const double* masses, const double* kappa, C* qg, C* ql, C* energy, std::uint64_t* status)
{
  const auto k = layout.laplacian_lanes;
  for (std::size_t e = 0; e < k; ++e)
  {
    const auto mu = masses[layout.target_electron_begin + e];
    auto ratio = add(l[e], make_cuDoubleComplex(cusp[1 + 3 * k + e], 0));
    if (!device_math::isFiniteBinary64(mu) || !(mu > 0) || !finite(ratio)) { fail(status); return; }
    ql[e] = make_cuDoubleComplex(-0.5 * mu, 0);
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
      const auto lane = 3 * e + axis;
      const auto total = add(g[lane], make_cuDoubleComplex(cusp[1 + lane], kappa[axis]));
      const auto square = QMC_PF_MUL(total, total);
      qg[lane] = scale(total, -mu);
      ratio = add(ratio, square);
      if (!finite(total) || !finite(square) || !finite(qg[lane]) || !finite(ratio)) { fail(status); return; }
    }
    *energy = add(*energy, scale(ratio, -0.5 * mu));
    if (!finite(*energy)) { fail(status); return; }
  }
}

/// Pack cusp once per real objective; physical lambdaL has zero imaginary component.
__global__ void cuspSeed(std::size_t k, const C* qg, const C* ql, std::size_t selected,
    double* seed, std::uint64_t* status)
{
  seed[0] = 0;
  for (std::size_t i = 0; i < 3 * k; ++i)
  {
    if (!finite(qg[i])) { fail(status); return; }
    seed[1 + i] = component(qg[i], selected);
  }
  for (std::size_t i = 0; i < k; ++i)
  {
    if (!finite(ql[i])) { fail(status); return; }
    seed[1 + 3 * k + i] = component(ql[i], selected);
  }
}

/// Add one completed realP to its selected complex component; reject a late sum overflow.
__global__ void packResponse(std::size_t p, const double* real, std::size_t selected,
    C* output, std::uint64_t* status)
{
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= p) return;
  const auto term = selected ? make_cuDoubleComplex(0, real[i]) : make_cuDoubleComplex(real[i], 0);
  output[i] = add(output[i], term);
  if (!finite(term) || !finite(output[i])) fail(status);
}

/// Only packed B1 FULL/compact-K leaves are admitted; numeric extents are checked first.
void validate(OpenOrbitalJetLayout layout)
{
  validateOpenOrbitalJetLayout(layout);
  const auto n = layout.features.electron_count;
  const auto nn = spatial_detail::checkedProduct(n, n, "Complex kinetic matrix extent overflow");
  (void)spatial_detail::checkedProduct(nn, std::size_t(3), "Complex kinetic scratch extent overflow");
  if (layout.features.configuration_count != 1 || layout.features.mode != SpatialJetMode::FULL_VGL ||
      layout.orbitals.element_count != spatial_detail::checkedProduct(nn, layout.determinant_count, "Complex kinetic extent overflow") ||
      layout.orbital_row_stride != n || layout.orbital_determinant_stride != nn ||
      layout.orbitals.plane_stride != layout.orbitals.element_count || layout.features.plane_stride != layout.features.element_count ||
      layout.orbitals.configuration_stride != spatial_detail::checkedProduct(layout.orbitals.plane_count, layout.orbitals.element_count, "Complex kinetic extent overflow") ||
      layout.features.configuration_stride != spatial_detail::checkedProduct(layout.features.plane_count, layout.features.element_count, "Complex kinetic extent overflow"))
    throw std::invalid_argument("Complex kinetic leaves require packed B1 FULL/compact-K jets");
  (void)spatialJetSpanElements(layout.orbitals);
}

/// Return launch status without hiding caller-owned numerical admission synchronization.
Error launched()
{
#ifdef QMC_CUDA2HIP
  return hipPeekAtLastError();
#else
  return cudaPeekAtLastError();
#endif
}
} // namespace

/// Stateless root leaf; no allocations or inverse mutation, and no domain fallback.
Error launchComplexKineticMatrixRoots(Stream stream, OpenOrbitalJetLayout layout,
    const ComplexValueChannel* channels, const ComplexValueResult* amplitude, const C* inverse,
    const C* orbitals, const C* gradient, const C* laplacian, const C* qg, const C* ql,
    C* roots, C* scratch, std::uint64_t* status)
{
  validate(layout);
  if (!channels || !amplitude || !inverse || !orbitals || !gradient || !laplacian || !qg || !ql || !roots || !scratch || !status)
    throw std::invalid_argument("Complex kinetic root storage is null");
  const void* reads[]{channels, amplitude, inverse, orbitals, gradient, laplacian, qg, ql};
  const void* outputs[]{roots, scratch, status};
  open_kinetic_reverse_detail::rejectAliasedStorage(reads, outputs, "Complex kinetic root storage aliases");
  if (cooperativeMatrixRootsRequested())
    matrixRootsCooperative<><<<1, 256, 0, stream>>>(layout, channels, amplitude, inverse, orbitals, gradient, laplacian, qg, ql, roots, scratch, status, nullptr);
  else
    matrixRoots<<<1, 1, 0, stream>>>(layout, channels, amplitude, inverse, orbitals, gradient, laplacian, qg, ql, roots, scratch, status);
  return launched();
}

/// Check all live byte spans before resetting private state or launching any work.
Error launchComplexKineticChannelRoots(Stream stream, OpenOrbitalJetLayout layout,
    const ComplexValueChannel* channels, const ComplexValueResult* amplitude, const C* inverse,
    const C* orbitals, const C* gradient, const C* laplacian, const C* qg, const C* ql,
    C* roots, C* scratch, std::size_t scratch_capacity, std::uint64_t* status, std::uint64_t* private_failure)
{
  validate(layout);
  const auto n = layout.features.electron_count, d = layout.determinant_count;
  if (n < 64 || n > 300 || d < 2 || d > 16)
    throw std::invalid_argument("Channel kinetic roots require Ne64..300 and D2..16");
  const auto mul = [](std::size_t a, std::size_t b) {
    return spatial_detail::checkedProduct(a, b, "Channel kinetic root extent overflow");
  };
  const auto m = layout.orbitals.element_count, k = layout.orbitals.laplacian_lanes;
  const auto required = mul(3, mul(d, mul(n, n)));
  if (scratch_capacity < required)
    throw std::length_error("Channel kinetic root scratch is too small");
  struct Span
  {
    const void* pointer;
    std::size_t bytes;
  };
  const Span reads[]{{channels, mul(d, sizeof(ComplexValueChannel))}, {amplitude, sizeof(ComplexValueResult)},
      {inverse, mul(m, sizeof(C))}, {orbitals, mul(mul(layout.orbitals.plane_count, m), sizeof(C))},
      {gradient, mul(mul(3, k), sizeof(C))}, {laplacian, mul(k, sizeof(C))},
      {qg, mul(mul(3, k), sizeof(C))}, {ql, mul(k, sizeof(C))}};
  const Span outputs[]{{roots, mul(mul(layout.orbitals.plane_count, m), sizeof(C))},
      {scratch, mul(required, sizeof(C))}, {status, sizeof(std::uint64_t)},
      {private_failure, sizeof(std::uint64_t)}};
  const auto address = [](Span span) {
    const auto first = reinterpret_cast<std::uintptr_t>(span.pointer);
    if (!span.pointer)
      throw std::invalid_argument("Channel kinetic root storage is null");
    if (span.bytes > std::numeric_limits<std::uintptr_t>::max() - first)
      throw std::overflow_error("Channel kinetic root address extent overflow");
    return first;
  };
  const auto overlap = [&](Span a, Span b) {
    const auto first = address(a), second = address(b);
    return first < second + b.bytes && second < first + a.bytes;
  };
  for (const auto& read : reads)
    (void)address(read);
  for (std::size_t i = 0; i < 4; ++i)
  {
    (void)address(outputs[i]);
    for (const auto& read : reads)
      if (overlap(outputs[i], read))
        throw std::invalid_argument("Channel kinetic root input/output spans overlap");
    for (std::size_t j = 0; j < i; ++j)
      if (overlap(outputs[i], outputs[j]))
        throw std::invalid_argument("Channel kinetic root writable spans overlap");
  }
#ifdef QMC_CUDA2HIP
  auto result = hipMemsetAsync(private_failure, 0xff, sizeof(std::uint64_t), stream);
#else
  auto result = cudaMemsetAsync(private_failure, 0xff, sizeof(std::uint64_t), stream);
#endif
  if (result != success)
    return result;
  matrixRootsCooperative<true><<<static_cast<unsigned>(d), 256, 0, stream>>>(layout, channels, amplitude,
      inverse, orbitals, gradient, laplacian, qg, ql, roots, scratch, private_failure, nullptr);
  result = launched();
  if (result != success)
    return result;
  matrixRootsCooperative<false, true><<<1, 256, 0, stream>>>(layout, channels, amplitude,
      inverse, orbitals, gradient, laplacian, qg, ql, roots, scratch, status, private_failure);
  return launched();
}

/// One selected real objective; all canonical intervals checked before its serial launch.
static void validateComplexKineticHeadReverse(Stream stream, ComplexKineticHeadLayout l,
    const periodic_geometry::Cell& cell, const double* positions, const double* nuclei, const double* p,
    const double* features, const double* envelope, const double* real, const double* imag,
    const C* roots, std::size_t selected, double* feature_seed, double* response, std::uint64_t* status)
{
  validate(l.real.forward);
  validateOpenOrbitalKineticReverseLayout(l.real);
  periodic_geometry::validateCell(cell);
  using namespace open_kinetic_reverse_detail;
  const auto span = openOrbitalBackflowSpan(l.real.forward), es = openOrbitalEnvelopeSpan(l.real.forward);
  const Interval intervals[]{checkedInterval(l.imaginary_up, span, l.real.response_count, "Imaginary head exceeds P"),
      checkedInterval(l.imaginary_down, span, l.real.response_count, "Imaginary head exceeds P"),
      {l.real.backflow_up_response_offset, l.real.backflow_up_response_offset + span},
      {l.real.backflow_down_response_offset, l.real.backflow_down_response_offset + span},
      {l.real.envelope_pi_up_response_offset, l.real.envelope_pi_up_response_offset + es},
      {l.real.envelope_pi_down_response_offset, l.real.envelope_pi_down_response_offset + es},
      {l.real.envelope_zeta_up_response_offset, l.real.envelope_zeta_up_response_offset + es},
      {l.real.envelope_zeta_down_response_offset, l.real.envelope_zeta_down_response_offset + es}};
  for (std::size_t i = 0; i < 8; ++i)
    for (std::size_t j = i + 1; j < 8; ++j)
      if (overlaps(intervals[i], intervals[j])) throw std::invalid_argument("Complex kinetic canonical intervals overlap");
  if (selected > 1 || !positions || !nuclei || !p || !features || !envelope || !real || !imag || !roots || !feature_seed || !response || !status)
    throw std::invalid_argument("Complex kinetic head inputs are invalid");
  const void* reads[]{positions, nuclei, p, features, envelope, real, imag, roots};
  const void* outputs[]{feature_seed, response, status};
  rejectAliasedStorage(reads, outputs, "Complex kinetic head storage aliases");
}

/// Strict cold capture is shared by enclosing cost and leaf preparation.
bool complexKineticHeadParallelRequested()
{
  static const bool requested = [] {
    const char* value = std::getenv("PSIFORMER_GPU_COMPLEX_KINETIC_HEAD_PARALLEL");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::invalid_argument("PSIFORMER_GPU_COMPLEX_KINETIC_HEAD_PARALLEL must be 0 or 1");
  }();
  return requested;
}

/// Default entry retains its original single-thread arithmetic and destination semantics.
Error launchComplexKineticHeadReverse(Stream stream, ComplexKineticHeadLayout l,
    const periodic_geometry::Cell& cell, const double* positions, const double* nuclei, const double* p,
    const double* features, const double* envelope, const double* real, const double* imag,
    const C* roots, std::size_t selected, double* feature_seed, double* response, std::uint64_t* status)
{
  validateComplexKineticHeadReverse(stream, l, cell, positions, nuclei, p, features, envelope, real, imag,
      roots, selected, feature_seed, response, status);
  headReverse<<<1, 1, 0, stream>>>(l, cell, positions, nuclei, p, features, envelope, real, imag, roots, selected, feature_seed, response, status);
  return launched();
}

/// Independent output reductions use complete private storage and one ordered publish/replay boundary.
Error launchComplexKineticHeadReverseParallel(Stream stream, ComplexKineticHeadLayout l,
    const periodic_geometry::Cell& cell, const double* positions, const double* nuclei, const double* p,
    const double* features, const double* envelope, const double* real, const double* imag,
    const C* roots, std::size_t selected, double* feature_seed, double* response, std::uint64_t* status,
    double* private_features, std::size_t feature_capacity, double* private_response,
    std::size_t response_capacity, std::uint64_t* private_status)
{
  validateComplexKineticHeadReverse(stream, l, cell, positions, nuclei, p, features, envelope, real, imag,
      roots, selected, feature_seed, response, status);
  const auto& layout = l.real.forward;
  const auto n = layout.features.electron_count, d = layout.determinant_count;
  if (n < 64 || n > 300 || d > 16 || cell.separate_ion_cell)
    return launchComplexKineticHeadReverse(stream, l, cell, positions, nuclei, p, features, envelope, real, imag,
        roots, selected, feature_seed, response, status);
  const auto mul = [](auto a, auto b) { return spatial_detail::checkedProduct(a, b, "Parallel kinetic head extent overflow"); };
  const auto fcount = mul(layout.features.element_count, layout.orbitals.plane_count);
  const auto pcount = l.real.response_count;
  const auto weight_count = mul(2, mul(layout.feature_width, mul(d, n)));
  const auto feature_count = mul(layout.orbitals.plane_count, mul(n, layout.feature_width));
  const auto envelope_count = mul(2, mul(d, mul(n, layout.nucleus_count)));
  const auto blocks = [](std::size_t count) {
    if (!count || (count - 1) / 128 >= std::size_t(std::numeric_limits<int>::max()))
      throw std::overflow_error("Parallel kinetic head grid exceeds admission");
    return static_cast<unsigned>((count - 1) / 128 + 1);
  };
  const auto publish_blocks = blocks(std::max(pcount, fcount));
  const auto wb = blocks(weight_count), fb = blocks(feature_count), eb = blocks(envelope_count);
  if (!private_features || !private_response || !private_status || feature_capacity < fcount || response_capacity < pcount)
    throw std::invalid_argument("Parallel kinetic head private capacity is insufficient");
  const void* reads[]{positions, nuclei, p, features, envelope, real, imag, roots, feature_seed, response, status};
  const void* outputs[]{private_features, private_response, private_status};
  open_kinetic_reverse_detail::rejectAliasedStorage(reads, outputs, "Parallel kinetic private storage aliases");
  beginHeadReverse<<<publish_blocks, 128, 0, stream>>>(pcount, fcount, response, feature_seed,
      private_response, private_features, private_status);
  if (auto error = launched(); error != success) return error;
  parallelHeadReverse<1><<<wb, 128, 0, stream>>>(weight_count, l, cell, positions, nuclei, p, features,
      envelope, real, imag, roots, selected, private_features, private_response, private_status);
  if (auto error = launched(); error != success) return error;
  parallelHeadReverse<2><<<fb, 128, 0, stream>>>(feature_count, l, cell, positions, nuclei, p, features,
      envelope, real, imag, roots, selected, private_features, private_response, private_status);
  if (auto error = launched(); error != success) return error;
  parallelHeadReverse<3><<<eb, 128, 0, stream>>>(envelope_count, l, cell, positions, nuclei, p, features,
      envelope, real, imag, roots, selected, private_features, private_response, private_status);
  if (auto error = launched(); error != success) return error;
  finishHeadReverse<<<publish_blocks, 128, 0, stream>>>(pcount, fcount, l, cell, positions, nuclei, p, features,
      envelope, real, imag, roots, selected, feature_seed, response, status,
      private_features, private_response, private_status);
  return launched();
}

/// Prepared caller supplies disjoint live buffers; no scratch or allocation is hidden here.
Error launchComplexKineticCoefficients(Stream stream, SpatialJetLayout layout, const C* g, const C* l,
    const double* cusp, const double* masses, const double* kappa, C* qg, C* ql, C* energy, std::uint64_t* status)
{
  validateSpatialJetLayout(layout);
  if (layout.mode != SpatialJetMode::FULL_VGL || layout.element_count != 1 || layout.configuration_count != 1 ||
      layout.plane_stride != 1 || layout.configuration_stride != layout.plane_count ||
      !g || !l || !cusp || !masses || !kappa || !qg || !ql || !energy || !status)
    throw std::invalid_argument("Complex kinetic physical root inputs invalid");
  coefficients<<<1, 1, 0, stream>>>(layout, g, l, cusp, masses, kappa, qg, ql, energy, status);
  return launched();
}

/// Select a real cusp objective without rerunning physical coefficients or energy.
Error launchComplexKineticCuspSeed(Stream stream, std::size_t k, const C* qg, const C* ql,
    std::size_t selected, double* seed, std::uint64_t* status)
{
  if (!k || selected > 1 || !qg || !ql || !seed || !status)
    throw std::invalid_argument("Complex kinetic cusp seed inputs invalid");
  cuspSeed<<<1, 1, 0, stream>>>(k, qg, ql, selected, seed, status);
  return launched();
}

/// Consume immediately admitted oneP; finite accumulation does not publish a partial request.
Error launchComplexKineticPack(Stream stream, std::size_t p, const double* real,
    std::size_t selected, C* output, std::uint64_t* status)
{
  if (!p || (p - 1) / 128 >= std::size_t(std::numeric_limits<int>::max()) || selected > 1 || !real || !output || !status)
    throw std::invalid_argument("Complex kinetic P packing inputs invalid");
  packResponse<<<static_cast<unsigned>((p - 1) / 128 + 1), 128, 0, stream>>>(p, real, selected, output, status);
  return launched();
}
} // namespace qmcplusplus::psiformer::device
#undef QMC_PF_MUL
#undef QMC_PF_DIV
